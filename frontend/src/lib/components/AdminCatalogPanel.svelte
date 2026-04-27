<script>
	import { apiBaseUrl } from '$lib/config/env';

	export let token = null;

	const emptyForm = {
		id: '',
		class_name: '',
		display_name: '',
		brand: '',
		model: '',
		colorway: '',
		sku: '',
		price_eur: '',
		retail_price: '',
		currency: '',
		release_year: '',
		release_date: '',
		gender: '',
		category: '',
		source: '',
		source_url: '',
		description: '',
		metadata_json: ''
	};

	let loading = false;
	let saving = false;
	let importing = false;
	let exporting = false;
	let error = '';
	let success = '';
	let items = [];
	let search = '';
	let filteredItems = [];
	let checkpointOnly = false;
	let selectedId = '';
	let form = { ...emptyForm };
	let importFile = null;
	let importResult = null;
	let importInput;
	let loadedForToken = '';
	let showImportExample = false;

	function filterItems(itemsToFilter, queryText) {
		const query = queryText.trim().toLowerCase();
		if (!query) {
			return itemsToFilter;
		}

		return itemsToFilter.filter((item) => {
			const fields = [
				item.class_name,
				item.display_name,
				item.brand,
				item.model,
				item.colorway,
				item.sku
			];
			return fields.some((value) => value && String(value).toLowerCase().includes(query));
		});
	}

	$: filteredItems = filterItems(items, search);

	function responseErrorMessage(data, fallback) {
		if (data && data.detail) {
			return data.detail;
		}
		return fallback;
	}

	function resetForm() {
		form = { ...emptyForm };
		selectedId = '';
	}

	function setSelectedItem(item) {
		selectedId = item.id;
		form = {
			id: item.id || '',
			class_name: item.class_name || '',
			display_name: item.display_name || '',
			brand: item.brand || '',
			model: item.model || '',
			colorway: item.colorway || '',
			sku: item.sku || '',
			price_eur: item.price_eur ?? '',
			retail_price: item.retail_price ?? '',
			currency: item.currency || '',
			release_year: item.release_year ?? '',
			release_date: item.release_date || '',
			gender: item.gender || '',
			category: item.category || '',
			source: item.source || '',
			source_url: item.source_url || '',
			description: item.description || '',
			metadata_json: item.metadata_json || ''
		};
	}

	async function loadProducts() {
		if (!token) {
			return;
		}
		loading = true;
		error = '';
		success = '';
		try {
			const params = new URLSearchParams({
				limit: '500',
				checkpoint_only: checkpointOnly ? 'true' : 'false'
			});
			const response = await fetch(`${apiBaseUrl}/admin/catalog-products?${params.toString()}`, {
				headers: {
					Authorization: `Bearer ${token}`
				}
			});
			const data = await response.json();
			if (!response.ok) {
				throw new Error(responseErrorMessage(data, 'Failed to load catalog products.'));
			}
			items = data.items || [];
			importResult = null;
			if (selectedId) {
				const selected = items.find((item) => item.id === selectedId);
				if (selected) {
					setSelectedItem(selected);
				}
			}
		} catch (err) {
			error = err?.message || 'Failed to load catalog products.';
		} finally {
			loading = false;
		}
	}

	async function saveSelected() {
		if (!token || !selectedId) {
			return;
		}
		saving = true;
		error = '';
		success = '';
		try {
			const payload = {
				display_name: form.display_name,
				brand: form.brand,
				model: form.model,
				colorway: form.colorway,
				sku: form.sku,
				price_eur: form.price_eur,
				retail_price: form.retail_price,
				currency: form.currency,
				release_year: form.release_year,
				release_date: form.release_date,
				gender: form.gender,
				category: form.category,
				source: form.source,
				source_url: form.source_url,
				description: form.description,
				metadata_json: form.metadata_json
			};
			const response = await fetch(`${apiBaseUrl}/admin/catalog-products/${selectedId}`, {
				method: 'PATCH',
				headers: {
					'Content-Type': 'application/json',
					Authorization: `Bearer ${token}`
				},
				body: JSON.stringify(payload)
			});
			const data = await response.json();
			if (!response.ok) {
				throw new Error(responseErrorMessage(data, 'Failed to update catalog product.'));
			}
			success = 'Catalog product updated.';
			items = items.map((item) => (item.id === data.id ? data : item));
			setSelectedItem(data);
		} catch (err) {
			error = err?.message || 'Failed to update catalog product.';
		} finally {
			saving = false;
		}
	}

	async function exportCsv() {
		if (!token) {
			return;
		}
		exporting = true;
		error = '';
		success = '';
		try {
			const response = await fetch(`${apiBaseUrl}/admin/catalog-products/export`, {
				headers: {
					Authorization: `Bearer ${token}`
				}
			});
			if (!response.ok) {
				let data = null;
				try {
					data = await response.json();
				} catch {
					data = null;
				}
				throw new Error(responseErrorMessage(data, 'Failed to export catalog CSV.'));
			}
			const blob = await response.blob();
			const url = URL.createObjectURL(blob);
			const anchor = document.createElement('a');
			anchor.href = url;
			anchor.download = 'catalog_products.csv';
			document.body.appendChild(anchor);
			anchor.click();
			anchor.remove();
			URL.revokeObjectURL(url);
			success = 'CSV export downloaded.';
			importResult = null;
		} catch (err) {
			error = err?.message || 'Failed to export catalog CSV.';
		} finally {
			exporting = false;
		}
	}

	async function importCsv() {
		if (!token || !importFile) {
			return;
		}
		importing = true;
		error = '';
		success = '';
		try {
			const formData = new FormData();
			formData.set('file', importFile);
			const response = await fetch(`${apiBaseUrl}/admin/catalog-products/import`, {
				method: 'POST',
				headers: {
					Authorization: `Bearer ${token}`
				},
				body: formData
			});
			const data = await response.json();
			if (!response.ok) {
				throw new Error(responseErrorMessage(data, 'Failed to import catalog CSV.'));
			}
			importResult = data;
			success = `CSV import updated ${data.updated} entries and skipped ${data.skipped?.length || 0}.`;
			importFile = null;
			await loadProducts();
		} catch (err) {
			error = err?.message || 'Failed to import catalog CSV.';
		} finally {
			importing = false;
		}
	}

	async function handleImportFileChange(event) {
		importFile = event.currentTarget.files?.[0] || null;
		if (!importFile) {
			return;
		}
		await importCsv();
		if (importInput) {
			importInput.value = '';
		}
	}

	function handleCheckpointToggle() {
		selectedId = '';
		loadProducts();
	}

	$: if (token && token !== loadedForToken) {
		loadedForToken = token;
		loadProducts();
	}
</script>

<section class="catalog-panel">
	<div class="section-heading">
		<div>
			<p class="eyebrow">Catalog Metadata</p>
			<h2>Browse and edit sneaker entries</h2>
			<p class="route-note">
				Search the sneaker catalog, edit metadata fields, export the current table to CSV,
				or import CSV updates for existing entries.
			</p>
		</div>
	</div>

	<div class="toolbar">
		<label class="search-field">
			<span>Search</span>
			<input bind:value={search} placeholder="Class name, display name, brand, SKU..." />
		</label>
		<label class="checkpoint-toggle">
			<input type="checkbox" bind:checked={checkpointOnly} onchange={handleCheckpointToggle} />
			<span>Only active checkpoint classes</span>
		</label>
		<div class="toolbar-actions">
			<button class="button ghost" type="button" disabled={exporting} onclick={exportCsv}>
				{exporting ? 'Exporting...' : 'Export CSV'}
			</button>
		</div>
	</div>

	<div class="import-row">
		<div class="import-field">
			<span>Import CSV updates</span>
			<input
				bind:this={importInput}
				type="file"
				accept=".csv,text/csv"
				onchange={handleImportFileChange}
			/>
			<div class="import-help">
				<p>
					The CSV must contain a header row. Each data row should identify an existing sneaker by
					<code>class_name</code>. Any additional columns are used as fields to update on that
					entry. If you edit an exported CSV, you may keep the <code>id</code> column, but it is
					not required.
				</p>
				<button
					class="example-toggle"
					type="button"
					onclick={() => (showImportExample = !showImportExample)}
				>
					{showImportExample ? 'Hide CSV example' : 'Show CSV example'}
				</button>
				{#if showImportExample}
					<pre>class_name,display_name,brand,model,price_eur,release_year,gender
Nike_Air_Force_1_Low,Nike Air Force 1 Low,Nike,Air Force 1 Low,100,1982,men
Nike_Dunk_Low,Nike Dunk Low,Nike,Dunk Low,120,1985,any</pre>
				{/if}
				<p>
					The safest workflow is to export the current CSV first, edit only the rows you want to
					change, and then import that edited file.
				</p>
			</div>
			{#if importFile && importing}
				<p class="selected-file">Selected: {importFile.name}</p>
			{/if}
		</div>
	</div>

	<div class="import-actions">
		<button class="button primary" type="button" disabled={importing} onclick={() => importInput?.click()}>
			{importing ? 'Importing...' : 'Import CSV'}
		</button>
	</div>

	{#if error}
		<p class="route-note route-error">{error}</p>
	{/if}
	{#if success}
		<p class="route-note route-success">{success}</p>
	{/if}
	{#if importResult?.skipped?.length}
		<div class="import-summary">
			<p class="route-note">
				Skipped rows:
			</p>
			<ul>
				{#each importResult.skipped.slice(0, 10) as item}
					<li>Row {item.row}: {item.reason}</li>
				{/each}
			</ul>
			{#if importResult.skipped.length > 10}
				<p class="route-note">Only the first 10 skipped rows are shown.</p>
			{/if}
		</div>
	{/if}

	<div class="catalog-layout">
		<section class="product-list">
			<div class="list-head">
				<strong>Entries</strong>
				{#if search.trim()}
					<span>{filteredItems.length} / {items.length}</span>
				{:else}
					<span>{items.length}</span>
				{/if}
			</div>
			{#if items.length === 0}
				<p class="route-note">No catalog entries loaded.</p>
			{:else if filteredItems.length === 0}
				<p class="route-note">No catalog entries match the current search.</p>
			{:else}
				<div class="list-stack">
					{#each filteredItems as item}
						<button
							type="button"
							class:selected={selectedId === item.id}
							class="product-row"
							onclick={() => setSelectedItem(item)}
						>
							<strong>{item.display_name}</strong>
							<span>{item.class_name}</span>
						</button>
					{/each}
				</div>
			{/if}
		</section>

		<section class="editor-card">
			{#if !selectedId}
				<p class="route-note">Select a catalog entry to edit its metadata.</p>
			{:else}
				<div class="editor-head">
					<div>
						<p class="eyebrow">Selected Entry</p>
						<h3>{form.display_name}</h3>
						<p class="route-note">{form.class_name}</p>
					</div>
				</div>

				<div class="field-grid">
					<label><span>Display name</span><input bind:value={form.display_name} /></label>
					<label><span>Brand</span><input bind:value={form.brand} /></label>
					<label><span>Model</span><input bind:value={form.model} /></label>
					<label><span>Colorway</span><input bind:value={form.colorway} /></label>
					<label><span>SKU</span><input bind:value={form.sku} /></label>
					<label><span>Category</span><input bind:value={form.category} /></label>
					<label><span>Price (EUR)</span><input bind:value={form.price_eur} /></label>
					<label><span>Retail price</span><input bind:value={form.retail_price} /></label>
					<label><span>Currency</span><input bind:value={form.currency} /></label>
					<label><span>Release year</span><input bind:value={form.release_year} /></label>
					<label><span>Release date</span><input bind:value={form.release_date} /></label>
					<label><span>Gender</span><input bind:value={form.gender} /></label>
					<label><span>Source</span><input bind:value={form.source} /></label>
					<label><span>Source URL</span><input bind:value={form.source_url} /></label>
				</div>

				<label class="notes-field">
					<span>Description</span>
					<textarea bind:value={form.description} rows="4"></textarea>
				</label>

				<label class="notes-field">
					<span>Metadata JSON</span>
					<textarea bind:value={form.metadata_json} rows="6" placeholder={'{"notes": "..."}'}></textarea>
				</label>

				<div class="action-row">
					<button class="button primary" type="button" disabled={saving} onclick={saveSelected}>
						{saving ? 'Saving...' : 'Save changes'}
					</button>
					<button class="button ghost" type="button" disabled={saving} onclick={resetForm}>
						Clear selection
					</button>
				</div>
			{/if}
		</section>
	</div>
</section>

<style>
	.catalog-panel {
		margin-top: 2rem;
		padding-top: 2rem;
		border-top: 1px solid rgba(77, 58, 46, 0.1);
	}

	.section-heading {
		display: flex;
		justify-content: space-between;
		gap: 1rem;
		align-items: start;
	}

	.route-note {
		margin: 0.35rem 0 0;
		color: var(--site-text-soft);
	}

	.route-error {
		color: var(--color-error);
	}

	.route-success {
		color: #2d6c47;
	}

	.selected-file {
		margin: 0.4rem 0 0;
		color: var(--site-text-muted);
		font-size: 0.88rem;
	}

	.import-field small {
		display: block;
		margin-top: 0.35rem;
		color: var(--site-text-muted);
		font-size: 0.8rem;
		line-height: 1.45;
	}

	.import-help {
		margin-top: 0.45rem;
		color: var(--site-text-muted);
		font-size: 0.84rem;
		line-height: 1.5;
	}

	.import-help p {
		margin: 0.35rem 0;
	}

	.example-toggle {
		margin-top: 0.35rem;
		padding: 0;
		border: none;
		background: none;
		color: var(--site-accent, #b8542e);
		font: inherit;
		font-weight: 700;
		cursor: pointer;
	}

	.example-toggle:hover {
		text-decoration: underline;
	}

	.import-help pre {
		margin: 0.5rem 0;
		padding: 0.8rem 0.9rem;
		overflow-x: auto;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 12px;
		background: rgba(255, 255, 255, 0.74);
		color: var(--site-text-soft);
		font-size: 0.78rem;
		line-height: 1.45;
	}

	.import-field code {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
	}

	.import-summary {
		margin-top: 0.75rem;
		padding: 0.85rem 0.95rem;
		border-radius: 16px;
		background: rgba(184, 132, 64, 0.08);
		border: 1px solid rgba(184, 132, 64, 0.12);
	}

	.import-summary ul {
		margin: 0.45rem 0 0;
		padding-left: 1.1rem;
		color: var(--site-text-soft);
	}

	.toolbar,
	.import-row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.9rem;
		align-items: end;
		margin-top: 1rem;
	}

	.import-actions {
		margin-top: 0.9rem;
	}

	.search-field {
		display: grid;
		gap: 0.35rem;
		min-width: min(100%, 420px);
	}

	.checkpoint-toggle {
		display: inline-flex;
		align-items: center;
		gap: 0.55rem;
		padding: 0.8rem 0.9rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 14px;
		background: rgba(255, 252, 247, 0.8);
	}

	.checkpoint-toggle input {
		width: auto;
		margin: 0;
	}

	.checkpoint-toggle span {
		color: var(--site-text-soft);
		font-weight: 700;
	}

	.search-field span,
	.import-field span,
	label span {
		display: block;
		color: var(--site-text-soft);
		font-weight: 700;
	}

	input,
	textarea {
		width: 100%;
		padding: 0.8rem 0.9rem;
		border: 1px solid rgba(77, 58, 46, 0.16);
		border-radius: 14px;
		background: rgba(255, 255, 255, 0.84);
		font: inherit;
	}

	.import-field input {
		display: none;
	}

	.toolbar-actions {
		display: flex;
		flex-wrap: wrap;
		gap: 0.7rem;
	}

	.catalog-layout {
		display: grid;
		grid-template-columns: minmax(260px, 340px) minmax(0, 1fr);
		gap: 1rem;
		margin-top: 1.2rem;
	}

	.product-list,
	.editor-card {
		padding: 1.1rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 22px;
		background: rgba(255, 255, 255, 0.76);
	}

	.list-head {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 0.8rem;
		margin-bottom: 0.85rem;
	}

	.list-stack {
		display: grid;
		gap: 0.55rem;
		max-height: 620px;
		overflow: auto;
	}

	.product-row {
		display: grid;
		gap: 0.2rem;
		padding: 0.8rem 0.9rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 16px;
		background: rgba(255, 255, 255, 0.72);
		text-align: left;
		cursor: pointer;
	}

	.product-row.selected {
		background: var(--site-selected);
		box-shadow: inset 0 0 0 1px rgba(212, 87, 46, 0.16);
	}

	.product-row span {
		color: var(--site-text-muted);
		font-size: 0.8rem;
		word-break: break-word;
	}

	.field-grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.9rem;
		margin-top: 1rem;
	}

	.notes-field {
		display: block;
		margin-top: 0.9rem;
	}

	.action-row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.7rem;
		margin-top: 1rem;
	}

	@media (max-width: 900px) {
		.catalog-layout {
			grid-template-columns: 1fr;
		}
	}

	@media (max-width: 640px) {
		.field-grid {
			grid-template-columns: 1fr;
		}
	}
</style>
