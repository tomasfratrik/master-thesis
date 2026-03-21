<script>
	import { apiBaseUrl } from '$lib/config/env';

	export let token = null;

	let submitting = false;
	let error = '';
	let result = null;

	let brand = '';
	let displayName = '';
	let className = '';
	let notes = '';
	let referenceFiles = [];
	let testFiles = [];
	let previewFiles = [];
	let topK = 5;
	let skipPreprocess = false;

	function fileNames(files) {
		return Array.from(files || []).map((file) => file.name);
	}

	function prettyPercent(score) {
		if (score === null || score === undefined) return 'n/a';
		return `${(Number(score) * 100).toFixed(1)}%`;
	}

	function sourceLabel(source) {
		return source === 'preprocessed' ? 'Preprocessed' : 'Original image';
	}

	function clearForm() {
		brand = '';
		displayName = '';
		className = '';
		notes = '';
		referenceFiles = [];
		testFiles = [];
		previewFiles = [];
		topK = 5;
		skipPreprocess = false;
	}

	async function savePrototypeClass() {
		if (!token) return;
		submitting = true;
		error = '';
		result = null;

		try {
			const formData = new FormData();
			formData.set('brand', brand.trim());
			formData.set('display_name', displayName.trim());
			formData.set('class_name', className.trim());
			formData.set('notes', notes.trim());
			formData.set('top_k', String(topK));
			formData.set('skip_preprocess', skipPreprocess ? 'true' : 'false');
			for (const file of referenceFiles) formData.append('reference_files', file);
			for (const file of testFiles) formData.append('test_files', file);
			for (const file of previewFiles) formData.append('preview_files', file);

			const response = await fetch(`${apiBaseUrl}/admin/prototype-classes`, {
				method: 'POST',
				headers: {
					Authorization: `Bearer ${token}`
				},
				body: formData
			});
			const data = await response.json();
			if (!response.ok) {
				throw new Error(data?.detail || 'Failed to add prototype class.');
			}
			result = data;
			clearForm();
		} catch (err) {
			error = err.message || 'Failed to add prototype class.';
		} finally {
			submitting = false;
		}
	}
</script>

<section class="training-panel">
	<div class="section-heading">
		<div>
			<p class="eyebrow">Prototype Classes</p>
			<h2>Add a sneaker class without fine-tuning</h2>
			<p class="route-note">
				Upload reference images for the new sneaker, generate a prototype embedding immediately,
				and evaluate it on your uploaded test images before using it in matching.
			</p>
		</div>
	</div>

	<section class="training-form-card">
		<div class="field-grid">
			<label>
				<span>Brand</span>
				<input bind:value={brand} placeholder="New Balance" />
			</label>
			<label>
				<span>Display name</span>
				<input bind:value={displayName} placeholder="9060" />
			</label>
			<label>
				<span>Save as class name</span>
				<input bind:value={className} placeholder="New_Balance_9060" />
			</label>
			<label>
				<span>Evaluation top-K</span>
				<input bind:value={topK} type="number" min="1" max="10" />
			</label>
		</div>

		<label class="notes-field">
			<span>Notes</span>
			<textarea bind:value={notes} rows="3" placeholder="Optional notes about the added class."></textarea>
		</label>

		<label class="toggle-row">
			<input type="checkbox" bind:checked={skipPreprocess} />
			<div>
				<span>Skip preprocess</span>
				<small>Use the original uploaded images directly when building and testing the prototype.</small>
			</div>
		</label>

		<div class="upload-blocks">
			<label class="upload-field">
				<span>Reference images</span>
				<input
					type="file"
					accept="image/*"
					multiple
					onchange={(event) => (referenceFiles = Array.from(event.currentTarget.files || []))}
				/>
				<small>{referenceFiles.length} selected</small>
				{#if referenceFiles.length}
					<ul>{#each fileNames(referenceFiles) as name}<li>{name}</li>{/each}</ul>
				{/if}
			</label>
			<label class="upload-field">
				<span>Test images</span>
				<input
					type="file"
					accept="image/*"
					multiple
					onchange={(event) => (testFiles = Array.from(event.currentTarget.files || []))}
				/>
				<small>{testFiles.length} selected</small>
				{#if testFiles.length}
					<ul>{#each fileNames(testFiles) as name}<li>{name}</li>{/each}</ul>
				{/if}
			</label>
			<label class="upload-field">
				<span>Preview images</span>
				<input
					type="file"
					accept="image/*"
					multiple
					onchange={(event) => (previewFiles = Array.from(event.currentTarget.files || []))}
				/>
				<small>{previewFiles.length} selected</small>
				{#if previewFiles.length}
					<ul>{#each fileNames(previewFiles) as name}<li>{name}</li>{/each}</ul>
				{/if}
			</label>
		</div>

		<div class="action-row">
			<button
				class="button primary"
				type="button"
				disabled={submitting || !brand.trim() || !displayName.trim() || referenceFiles.length === 0}
				onclick={savePrototypeClass}
			>
				{submitting ? 'Saving class...' : 'Create prototype class'}
			</button>
		</div>

		{#if error}
			<p class="route-note route-error">{error}</p>
		{/if}
	</section>

	{#if result}
		<section class="jobs-card">
			<div class="jobs-head">
				<div>
					<p class="eyebrow">Latest Result</p>
					<h3>{result.display_name}</h3>
					<p class="route-note">{result.class_name}</p>
				</div>
				<span class="status-badge completed">Saved</span>
			</div>

			<div class="metric-grid">
				<div><span>Reference crops</span><strong>{result.reference_image_count}</strong></div>
				<div><span>Test images</span><strong>{result.evaluation.summary.test_image_count ?? 0}</strong></div>
				<div><span>Top-1 accuracy</span><strong>{prettyPercent(result.evaluation.summary.top1_accuracy)}</strong></div>
				<div><span>Top-{topK} accuracy</span><strong>{prettyPercent(result.evaluation.summary[`top${topK}_accuracy`])}</strong></div>
				<div><span>Preprocess</span><strong>{result.skip_preprocess ? 'Skipped' : 'Enabled'}</strong></div>
			</div>

			{#if result.warnings?.length}
				<div class="warning-stack">
					{#each result.warnings as warning}
						<p class="route-note warning-note">{warning.message}</p>
					{/each}
				</div>
			{/if}

			{#if result.processed_reference_images?.length}
				<div class="image-block">
					<p class="eyebrow">{result.skip_preprocess ? 'Reference images used' : 'Processed reference images'}</p>
					<div class="image-grid">
						{#each result.processed_reference_images as image}
							<figure class="image-card">
								<img src={image.data_url} alt={image.filename} />
								<figcaption>{sourceLabel(image.source)}</figcaption>
							</figure>
						{/each}
					</div>
				</div>
			{/if}

			{#if result.evaluation.results?.length}
				<div class="result-list">
					{#each result.evaluation.results as item, index}
						<article class="result-card">
							<div class="result-head">
								<div>
									<p class="result-rank">Test #{index + 1}</p>
									<h4>{item.input_filename}</h4>
								</div>
								<strong>{item.is_top1 ? 'Top-1 hit' : item.is_topk ? `Top-${topK} hit` : 'Miss'}</strong>
							</div>
							<p class="route-note">
								Predicted: {item.prediction.label} ({prettyPercent(item.prediction.score)})
							</p>
						</article>
					{/each}
				</div>
			{/if}
		</section>
	{/if}
</section>

<style>
	.training-panel {
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

	.training-form-card,
	.jobs-card {
		margin-top: 1.2rem;
		padding: 1.2rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 22px;
		background: rgba(255, 255, 255, 0.76);
	}

	.field-grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.9rem;
	}

	label span {
		display: block;
		margin-bottom: 0.35rem;
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

	.notes-field {
		display: block;
		margin-top: 0.9rem;
	}

	.toggle-row {
		display: flex;
		gap: 0.75rem;
		align-items: start;
		margin-top: 1rem;
		padding: 0.9rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 18px;
		background: rgba(255, 255, 255, 0.68);
	}

	.toggle-row input {
		width: auto;
		margin-top: 0.2rem;
		accent-color: rgb(17, 88, 122);
	}

	.toggle-row small {
		display: block;
		margin-top: 0.2rem;
		color: var(--site-text-muted);
	}

	.upload-blocks {
		display: grid;
		gap: 0.9rem;
		margin-top: 1rem;
	}

	.upload-field {
		display: block;
		padding: 0.9rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 18px;
		background: rgba(255, 255, 255, 0.68);
	}

	.upload-field input {
		margin: 0.45rem 0;
		padding: 0;
		border: none;
		background: transparent;
	}

	.upload-field small,
	.metric-grid span {
		display: block;
		color: var(--site-text-muted);
		font-size: 0.78rem;
		font-weight: 700;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.upload-field ul {
		margin: 0.55rem 0 0;
		padding-left: 1rem;
		color: var(--site-text-soft);
		font-size: 0.9rem;
	}

	.jobs-head,
	.result-head {
		display: flex;
		justify-content: space-between;
		gap: 0.8rem;
		align-items: start;
	}

	.route-note {
		margin: 0.35rem 0 0;
		color: var(--site-text-soft);
	}

	.route-error {
		color: var(--color-error);
	}

	.metric-grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.75rem;
		margin-top: 0.9rem;
	}

	.metric-grid div {
		padding: 0.75rem 0.85rem;
		border-radius: 16px;
		background: rgba(255, 255, 255, 0.78);
	}

	.metric-grid strong {
		display: block;
		margin-top: 0.25rem;
	}

	.image-block {
		margin-top: 1rem;
	}

	.image-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 0.8rem;
		margin-top: 0.75rem;
	}

	.image-card {
		margin: 0;
		width: 160px;
		padding: 0.65rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 18px;
		background: rgba(255, 255, 255, 0.76);
	}

	.image-card img {
		display: block;
		width: 100%;
		height: 130px;
		object-fit: contain;
		border-radius: 12px;
		background: rgba(239, 231, 218, 0.55);
	}

	.image-card figcaption {
		margin-top: 0.5rem;
		color: var(--site-text-muted);
		font-size: 0.78rem;
		font-weight: 700;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.warning-stack {
		display: grid;
		gap: 0.5rem;
		margin-top: 1rem;
	}

	.warning-note {
		padding: 0.8rem 0.9rem;
		border-radius: 14px;
		background: rgba(184, 132, 64, 0.08);
		border: 1px solid rgba(184, 132, 64, 0.12);
	}

	.result-list {
		display: grid;
		gap: 0.9rem;
		margin-top: 1rem;
	}

	.result-card {
		padding: 1rem 1.05rem;
		border-radius: 20px;
		border: 1px solid rgba(77, 58, 46, 0.12);
		background: rgba(255, 255, 255, 0.76);
	}

	.result-rank {
		margin: 0;
		color: var(--site-text-muted);
		font-size: 0.76rem;
		font-weight: 700;
		letter-spacing: 0.14em;
		text-transform: uppercase;
	}

	.status-badge {
		padding: 0.45rem 0.8rem;
		border-radius: 999px;
		background: rgba(53, 136, 94, 0.16);
		color: #2d6c47;
		font-size: 0.78rem;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	@media (max-width: 640px) {
		.field-grid,
		.metric-grid {
			grid-template-columns: 1fr;
		}
	}
</style>
