<script>
	import { onMount } from 'svelte';
	import { apiBaseUrl } from '$lib/config/env';

	let loading = true;
	let error = '';
	let items = [];
	let brands = [];
	let groups = {};
	let activeBrand = 'All';
	let searchQuery = '';
	let classSource = '';
	let checkpointName = '';

	function sneakersForActiveBrand() {
		if (activeBrand === 'All') {
			return items;
		}

		if (groups[activeBrand]) {
			return groups[activeBrand];
		}

		return [];
	}

	function supportedSneakersError(data) {
		if (data && data.detail) {
			return data.detail;
		}

		return 'Failed to load supported sneakers.';
	}

	function visibleItems() {
		const pool = sneakersForActiveBrand();
		const query = searchQuery.trim().toLowerCase();

		if (!query) {
			return pool;
		}

		return pool.filter(
			(item) =>
				item.label.toLowerCase().includes(query) || item.brand.toLowerCase().includes(query)
		);
	}

	onMount(async () => {
		try {
			const response = await fetch(`${apiBaseUrl}/supported-sneakers`);
			const data = await response.json();

			if (!response.ok) {
				throw new Error(supportedSneakersError(data));
			}

			if (data.items) {
				items = data.items;
			} else {
				items = [];
			}

			if (data.brands) {
				brands = data.brands;
			} else {
				brands = [];
			}

			if (data.groups) {
				groups = data.groups;
			} else {
				groups = {};
			}

			classSource = data.class_source || '';
			checkpointName = data.checkpoint_name || '';
		} catch (err) {
			if (err && err.message) {
				error = err.message;
			} else {
				error = 'Failed to load supported sneakers.';
			}
		} finally {
			loading = false;
		}
	});
</script>

<svelte:head>
	<title>Supported Sneakers | Sneaker Matcher</title>
</svelte:head>

<section class="page-grid">
	<section class="panel section-card">
		<h1>Supported Sneakers</h1>

		{#if loading}
			<p class="route-note">Loading supported sneakers...</p>
		{:else if error}
			<p class="route-note route-error">{error}</p>
		{:else}
			<div class="summary-row">
				<p class="route-subnote">
					Choose a brand family to narrow the matcher label space.
				</p>
			</div>
			{#if classSource === 'checkpoint' && checkpointName}
				<p class="route-subnote source-note">
					These supported sneakers are taken from the currently loaded checkpoint:
					<strong>{checkpointName}</strong>
				</p>
			{/if}
			<div class="search-row">
				<label class="search-field" for="supported-search">
					<span>Search sneakers</span>
					<input
						id="supported-search"
						type="search"
						bind:value={searchQuery}
						placeholder="Search by model or brand"
					/>
				</label>
				<p class="search-count">Showing {visibleItems().length} results</p>
			</div>
			<div class="brand-row">
				<button
					type="button"
					class:active={activeBrand === 'All'}
					class="brand-chip"
					onclick={() => (activeBrand = 'All')}
				>
					All Brands
					<span>{items.length}</span>
				</button>
				{#each brands as item}
					<button
						type="button"
						class:active={activeBrand === item.brand}
						class="brand-chip"
						onclick={() => (activeBrand = item.brand)}
					>
						{item.brand}
						<span>{item.count}</span>
					</button>
				{/each}
			</div>
			<div class="name-grid">
				{#if visibleItems().length === 0}
					<p class="empty-state">No sneakers match the current brand and search filter.</p>
				{:else}
					{#each visibleItems() as item}
						<div class="name-card">
							<p>{item.label}</p>
							<span>{item.brand}</span>
						</div>
					{/each}
				{/if}
			</div>
		{/if}
	</section>
</section>

<style>
	.route-note {
		margin: 0;
		color: var(--site-text-soft);
	}

	.route-error {
		color: var(--color-error);
	}

	.route-subnote {
		margin: 0;
		color: var(--site-text-muted);
		font-size: 0.95rem;
	}

	.summary-row {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		justify-content: space-between;
		gap: 0.8rem 1.2rem;
		margin-top: 1rem;
	}

	.source-note {
		margin-top: 0.7rem;
	}

	.source-note strong {
		color: var(--site-text-soft);
		font-weight: 700;
	}

	.search-row {
		display: flex;
		flex-wrap: wrap;
		align-items: end;
		justify-content: space-between;
		gap: 0.9rem 1.2rem;
		margin-top: 1rem;
	}

	.search-field {
		display: grid;
		gap: 0.45rem;
		min-width: min(100%, 420px);
		color: var(--site-text-soft);
		font-size: 0.85rem;
		font-weight: 700;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.search-field input {
		width: 100%;
		padding: 0.9rem 1rem;
		border: 1px solid rgba(77, 58, 46, 0.14);
		border-radius: 16px;
		background: rgba(255, 255, 255, 0.8);
		color: var(--site-text);
		font: inherit;
		font-size: 1rem;
		letter-spacing: normal;
		text-transform: none;
		outline: none;
		transition:
			border-color 140ms ease,
			box-shadow 140ms ease;
	}

	.search-field input:focus {
		border-color: rgba(212, 87, 46, 0.35);
		box-shadow: 0 0 0 4px rgba(212, 87, 46, 0.08);
	}

	.search-count {
		margin: 0;
		color: var(--site-text-muted);
		font-size: 0.92rem;
	}

	.brand-row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.7rem;
		margin-top: 1rem;
	}

	.brand-chip {
		display: inline-flex;
		align-items: center;
		gap: 0.55rem;
		padding: 0.75rem 1rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 999px;
		background: rgba(255, 255, 255, 0.72);
		color: var(--site-text-soft);
		font: inherit;
		font-weight: 700;
		font-size: 0.83rem;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		cursor: pointer;
		transition:
			transform 140ms ease,
			background 140ms ease,
			box-shadow 140ms ease;
	}

	.brand-chip:hover {
		transform: translateY(-1px);
	}

	.brand-chip.active {
		background: var(--site-selected);
		color: var(--site-text);
		box-shadow: 0 10px 22px rgba(77, 51, 25, 0.08);
	}

	.brand-chip span {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		min-width: 1.8rem;
		height: 1.8rem;
		padding: 0 0.4rem;
		border-radius: 999px;
		background: rgba(39, 31, 26, 0.08);
		font-size: 0.8rem;
	}

	.name-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
		gap: 0.8rem;
		margin-top: 1.2rem;
	}

	.empty-state {
		grid-column: 1 / -1;
		margin: 0;
		padding: 1rem 1.1rem;
		border: 1px dashed rgba(77, 58, 46, 0.18);
		border-radius: 18px;
		background: rgba(255, 255, 255, 0.56);
		color: var(--site-text-muted);
	}

	.name-card {
		padding: 0.9rem 1rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 16px;
		background: rgba(255, 255, 255, 0.78);
		color: var(--site-text);
	}

	.name-card p {
		margin: 0;
		font-weight: 700;
	}

	.name-card span {
		display: inline-block;
		margin-top: 0.35rem;
		color: var(--site-text-muted);
		font-size: 0.84rem;
		font-weight: 700;
		letter-spacing: 0.1em;
		text-transform: uppercase;
	}
</style>
