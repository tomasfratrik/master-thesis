<script>
	import { apiBaseUrl } from '$lib/config/env';

	const aggregationOptions = [
		{ value: 'logit_mean', label: 'Logit mean', note: 'Best default for one sneaker with multiple views.' },
		{ value: 'embedding_mean', label: 'Embedding mean', note: 'Average image embeddings before classification.' },
		{ value: 'prob_mean', label: 'Probability mean', note: 'Average per-image class probabilities.' }
	];

	let mode = 'grouped';
	let aggregation = 'logit_mean';
	let selectedFiles = [];
	let filePreviews = [];
	let loading = false;
	let error = '';
	let response = null;
	let expandedPreviewKeys = new Set();
	let expandedTopKKeys = new Set();

	function syncPreviews(files) {
		for (const preview of filePreviews) {
			URL.revokeObjectURL(preview.url);
		}
		filePreviews = files.map((file) => ({
			name: file.name,
			url: URL.createObjectURL(file)
		}));
	}

	function updateFiles(list) {
		selectedFiles = Array.from(list || []);
		syncPreviews(selectedFiles);
		response = null;
		error = '';
		expandedPreviewKeys = new Set();
		expandedTopKKeys = new Set();
	}

	function toggleMode(nextMode) {
		mode = nextMode;
		response = null;
		error = '';
		expandedPreviewKeys = new Set();
		expandedTopKKeys = new Set();
	}

	function prettyPercent(score) {
		return `${(score * 100).toFixed(1)}%`;
	}

	function previewUrl(url) {
		return url?.startsWith('http') ? url : `${apiBaseUrl}${url}`;
	}

	function previewKey(prefix, index) {
		return `${prefix}-${index}`;
	}

	function warningText(warning) {
		if (warning.code === 'preprocess_multiple_crops_detected' && mode === 'grouped') {
			return `${warning.message} In One Sneaker mode, all detected crops are combined, so keep only the target sneaker visible from different angles. Switch to Multiple Sneakers if you want one result per detected crop.`;
		}
		return warning.message;
	}

	function togglePreview(key) {
		const next = new Set(expandedPreviewKeys);
		if (next.has(key)) {
			next.delete(key);
		} else {
			next.add(key);
		}
		expandedPreviewKeys = next;
	}

	function toggleTopK(key) {
		const next = new Set(expandedTopKKeys);
		if (next.has(key)) {
			next.delete(key);
		} else {
			next.add(key);
		}
		expandedTopKKeys = next;
	}

	async function analyze() {
		if (selectedFiles.length === 0) {
			error = 'Select at least one image.';
			return;
		}

		loading = true;
		error = '';
		response = null;
		expandedPreviewKeys = new Set();
		expandedTopKKeys = new Set();

		try {
			const formData = new FormData();
			for (const file of selectedFiles) {
				formData.append('files', file);
			}

			const params = new URLSearchParams({
				mode,
				top_k: '5'
			});
			if (mode === 'grouped') {
				params.set('aggregation', aggregation);
			}

			const res = await fetch(`${apiBaseUrl}/analyze?${params.toString()}`, {
				method: 'POST',
				body: formData
			});
			const data = await res.json();
			if (!res.ok) {
				throw new Error(data?.detail || 'Analysis failed.');
			}
			response = data;
		} catch (err) {
			error = err.message || 'Analysis failed.';
		} finally {
			loading = false;
		}
	}
</script>

<svelte:head>
	<title>Analyze | Sneaker Matcher</title>
</svelte:head>

<section class="page-grid">
	<div class="hero-grid analyze-grid">
		<section class="panel hero-copy">
			<p class="eyebrow">Visual Analysis</p>
			<h1>Upload sneaker photos and inspect the top matches.</h1>
			<p class="lede">
				Choose whether the uploaded images represent one sneaker with multiple views, or several
				separate sneakers that should each be analyzed on their own.
			</p>

			<div class="mode-tabs" role="tablist" aria-label="Analysis mode">
				<button
					type="button"
					role="tab"
					class:active={mode === 'grouped'}
					class="mode-tab"
					onclick={() => toggleMode('grouped')}
				>
					One Sneaker
				</button>
				<button
					type="button"
					role="tab"
					class:active={mode === 'per_image'}
					class="mode-tab"
					onclick={() => toggleMode('per_image')}
				>
					Multiple Sneakers
				</button>
			</div>

			<p class="mode-note">
				{#if mode === 'grouped'}
					All uploaded images will be treated as different views of the same sneaker.
				{:else}
					Each uploaded image will be analyzed separately, and the system will try to find all sneakers visible in each image.
				{/if}
			</p>

			<label class="upload-zone" for="analyze-files">
				<input
					id="analyze-files"
					type="file"
					accept="image/*"
					multiple
					onchange={(event) => updateFiles(event.currentTarget.files)}
				/>
				<div>
					<strong>Drop or browse sneaker photos</strong>
					<span>
						Use side, top, heel, or detail views. Upload multiple files for the chosen mode.
					</span>
				</div>
			</label>

			{#if filePreviews.length > 0}
				<div class="upload-preview-grid">
					{#each filePreviews as file}
						<figure class="upload-preview-card">
							<img src={file.url} alt={file.name} />
							<figcaption>{file.name}</figcaption>
						</figure>
					{/each}
				</div>
			{/if}

			{#if mode === 'grouped'}
				<div class="aggregation-block">
					<p class="eyebrow">Aggregation</p>
					<div class="aggregation-grid">
						{#each aggregationOptions as option}
							<button
								type="button"
								class:active={aggregation === option.value}
								class="aggregation-card"
								onclick={() => (aggregation = option.value)}
							>
								<strong>{option.label}</strong>
								<span>{option.note}</span>
							</button>
						{/each}
					</div>
				</div>
			{/if}

			<div class="cta-row">
				<button class="button primary" type="button" disabled={loading} onclick={analyze}>
					{loading ? 'Analyzing...' : 'Analyze Photos'}
				</button>
				{#if selectedFiles.length > 0}
					<button class="button ghost" type="button" disabled={loading} onclick={() => updateFiles([])}>
						Clear
					</button>
				{/if}
			</div>

			{#if error}
				<div class="alert alert-error/10 analyze-error">
					<span>{error}</span>
				</div>
			{/if}
		</section>

		<section class="panel section-card status-panel">
			<p class="eyebrow">Request Summary</p>
			<div class="summary-stack">
				<div class="summary-line">
					<span>Mode</span>
					<strong>{mode === 'grouped' ? 'One Sneaker' : 'Multiple Sneakers'}</strong>
				</div>
				<div class="summary-line">
					<span>Files selected</span>
					<strong>{selectedFiles.length}</strong>
				</div>
				{#if mode === 'grouped'}
					<div class="summary-line">
						<span>Aggregation</span>
						<strong>{aggregation.replace('_', ' ')}</strong>
					</div>
				{/if}
				<div class="summary-line">
					<span>Top-K</span>
					<strong>5</strong>
				</div>
			</div>
			{#if loading}
				<div class="loading-card">
					<span class="loading loading-spinner loading-lg text-info"></span>
					<p>Preprocessing images and running prediction.</p>
				</div>
			{/if}
		</section>
	</div>

	{#if response}
		<section class="panel section-card">
			<p class="eyebrow">Results</p>
			{#if response.warnings?.length}
				<div class="warning-stack">
					{#each response.warnings as warning}
						<div class="alert alert-warning/10 warning-alert">
							<span>{warningText(warning)}</span>
						</div>
					{/each}
				</div>
			{/if}
			{#if response.mode === 'grouped'}
				{@const result = response.result}
				{#if response.processed_images?.length}
					<div class="analyzed-block">
						<p class="eyebrow">Analyzed Images</p>
						<div class="analyzed-grid">
							{#each response.processed_images as image}
								<figure class="analyzed-card">
									<img src={image.data_url} alt={image.filename} />
									<figcaption>
										<strong>{image.filename}</strong>
										<span>{image.source}</span>
									</figcaption>
								</figure>
							{/each}
						</div>
					</div>
				{/if}

				<div class="winner-card">
					<div>
						<p class="winner-label">Top match</p>
						<h2>{result.label}</h2>
						<p class="winner-meta">
							Score {prettyPercent(result.score)}
							{#if result.margin_vs_second !== undefined}
								<span>Margin {prettyPercent(result.margin_vs_second)}</span>
							{/if}
						</p>
					</div>
					<div class="winner-badges">
						<span>{response.query_image_count} photos</span>
						<span>{response.processed_image_count} crops</span>
						<span>{result.aggregation}</span>
					</div>
				</div>

				<div class="topk-toggle-row">
					<button class="button ghost compact" type="button" onclick={() => toggleTopK('grouped-topk')}>
						{expandedTopKKeys.has('grouped-topk') ? 'Hide top 5 results' : 'Show top 5 results'}
					</button>
				</div>
				{#if expandedTopKKeys.has('grouped-topk')}
					<div class="result-list">
						{#each result.top_k as candidate, index}
							{@const key = previewKey('grouped', index)}
							<article class="result-card">
								<div class="result-head">
									<div>
										<p class="result-rank">#{index + 1}</p>
										<h3>{candidate.label}</h3>
									</div>
									<strong>{prettyPercent(candidate.score)}</strong>
								</div>
								<progress class="progress progress-info w-full" value={candidate.score * 100} max="100"></progress>
								<div class="result-actions">
									<button class="button ghost compact" type="button" onclick={() => togglePreview(key)}>
										{expandedPreviewKeys.has(key) ? 'Hide previews' : 'Show previews'}
									</button>
								</div>
								{#if expandedPreviewKeys.has(key)}
									<div class="preview-grid">
										{#if candidate.preview_urls?.length}
											{#each candidate.preview_urls as url}
												<img src={previewUrl(url)} alt={candidate.label} />
											{/each}
										{:else}
											<p class="empty-inline">No preview images available.</p>
										{/if}
									</div>
								{/if}
							</article>
						{/each}
					</div>
				{/if}
			{:else}
				<div class="per-image-stack">
					{#each response.results as result, resultIndex}
						<section class="per-image-card">
							<div class="winner-card">
								{#if result.processed_image}
									<div class="winner-image">
										<img src={result.processed_image.data_url} alt={result.query_filename} />
									</div>
								{/if}
								<div>
									<p class="winner-label">{result.query_filename}</p>
									<p class="processed-name">{result.processed_filename}</p>
									<h2>{result.label}</h2>
									<p class="winner-meta">Score {prettyPercent(result.score)}</p>
								</div>
								<div class="winner-badges">
									<span>{result.prepared_source}</span>
								</div>
							</div>

							<div class="topk-toggle-row">
								<button
									class="button ghost compact"
									type="button"
									onclick={() => toggleTopK(`per-topk-${resultIndex}`)}
								>
									{expandedTopKKeys.has(`per-topk-${resultIndex}`)
										? 'Hide top 5 results'
										: 'Show top 5 results'}
								</button>
							</div>
							{#if expandedTopKKeys.has(`per-topk-${resultIndex}`)}
								<div class="result-list compact-list">
									{#each result.top_k as candidate, index}
										{@const key = previewKey(`per-${resultIndex}`, index)}
										<article class="result-card">
											<div class="result-head">
												<div>
													<p class="result-rank">#{index + 1}</p>
													<h3>{candidate.label}</h3>
												</div>
												<strong>{prettyPercent(candidate.score)}</strong>
											</div>
											<progress class="progress progress-info w-full" value={candidate.score * 100} max="100"></progress>
											<div class="result-actions">
												<button class="button ghost compact" type="button" onclick={() => togglePreview(key)}>
													{expandedPreviewKeys.has(key) ? 'Hide previews' : 'Show previews'}
												</button>
											</div>
											{#if expandedPreviewKeys.has(key)}
												<div class="preview-grid">
													{#if candidate.preview_urls?.length}
														{#each candidate.preview_urls as url}
															<img src={previewUrl(url)} alt={candidate.label} />
														{/each}
													{:else}
														<p class="empty-inline">No preview images available.</p>
													{/if}
												</div>
											{/if}
										</article>
									{/each}
								</div>
							{/if}
						</section>
					{/each}
				</div>
			{/if}
		</section>
	{/if}
</section>

<style>
	.analyze-grid {
		grid-template-columns: minmax(0, 1.3fr) minmax(280px, 360px);
	}

	.mode-tabs {
		display: flex;
		gap: 0.75rem;
		margin-top: 1.3rem;
	}

	.mode-tab {
		padding: 0.85rem 1rem;
		border: 1px solid rgba(77, 58, 46, 0.14);
		border-radius: 999px;
		background: rgba(255, 255, 255, 0.72);
		color: var(--site-text-soft);
		font: inherit;
		font-weight: 700;
		cursor: pointer;
		transition: background 140ms ease, transform 140ms ease, box-shadow 140ms ease;
	}

	.mode-tab.active {
		background: var(--site-selected);
		color: var(--site-text);
		box-shadow: 0 10px 22px rgba(77, 51, 25, 0.08);
	}

	.mode-note {
		margin: 0.9rem 0 0;
		color: var(--site-text-muted);
	}

	.upload-zone {
		display: block;
		margin-top: 1.25rem;
		padding: 1.2rem;
		border: 1.5px dashed rgba(17, 88, 122, 0.35);
		border-radius: 24px;
		background:
			linear-gradient(135deg, rgba(255, 255, 255, 0.64), rgba(242, 231, 218, 0.92)),
			repeating-linear-gradient(
				-45deg,
				rgba(17, 88, 122, 0.03) 0,
				rgba(17, 88, 122, 0.03) 12px,
				transparent 12px,
				transparent 24px
			);
		cursor: pointer;
	}

	.upload-zone input {
		display: none;
	}

	.upload-zone strong {
		display: block;
		font-size: 1.15rem;
	}

	.upload-zone span {
		display: block;
		margin-top: 0.45rem;
		color: var(--site-text-soft);
		line-height: 1.5;
	}

	.upload-preview-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
		gap: 0.8rem;
		margin-top: 1rem;
	}

	.upload-preview-card {
		margin: 0;
		padding: 0.65rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 18px;
		background: rgba(255, 255, 255, 0.76);
	}

	.upload-preview-card img {
		display: block;
		width: 100%;
		height: 110px;
		object-fit: cover;
		border-radius: 12px;
	}

	.upload-preview-card figcaption {
		margin-top: 0.55rem;
		color: var(--site-text-muted);
		font-size: 0.8rem;
		word-break: break-word;
	}

	.aggregation-block {
		margin-top: 1.35rem;
	}

	.aggregation-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
		gap: 0.8rem;
		margin-top: 0.8rem;
	}

	.aggregation-card {
		display: grid;
		gap: 0.35rem;
		padding: 1rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 18px;
		background: rgba(255, 255, 255, 0.72);
		text-align: left;
		cursor: pointer;
	}

	.aggregation-card.active {
		background: var(--site-selected);
	}

	.aggregation-card span {
		color: var(--site-text-muted);
		font-size: 0.88rem;
		line-height: 1.45;
	}

	.status-panel {
		padding: 1.8rem;
	}

	.summary-stack {
		display: grid;
		gap: 0.9rem;
	}

	.summary-line {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		padding-bottom: 0.75rem;
		border-bottom: 1px solid rgba(77, 58, 46, 0.08);
	}

	.summary-line span {
		color: var(--site-text-muted);
		font-size: 0.82rem;
		font-weight: 700;
		letter-spacing: 0.1em;
		text-transform: uppercase;
	}

	.loading-card {
		display: grid;
		place-items: center;
		gap: 0.85rem;
		margin-top: 1.5rem;
		padding: 1.4rem;
		border-radius: 22px;
		background: rgba(255, 255, 255, 0.64);
		text-align: center;
	}

	.analyze-error {
		margin-top: 1rem;
		border-radius: 18px;
	}

	.warning-stack {
		display: grid;
		gap: 0.75rem;
		margin-bottom: 1rem;
	}

	.warning-alert {
		border-radius: 18px;
	}

	.analyzed-block {
		margin-bottom: 1rem;
	}

	.analyzed-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
		gap: 0.8rem;
		margin-top: 0.75rem;
	}

	.analyzed-card {
		margin: 0;
		padding: 0.7rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 18px;
		background: rgba(255, 255, 255, 0.76);
	}

	.analyzed-card img,
	.winner-image img {
		display: block;
		width: 100%;
		height: 140px;
		object-fit: cover;
		border-radius: 12px;
	}

	.analyzed-card figcaption {
		display: grid;
		gap: 0.2rem;
		margin-top: 0.55rem;
	}

	.analyzed-card figcaption span {
		color: var(--site-text-muted);
		font-size: 0.78rem;
		font-weight: 700;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.winner-card {
		display: flex;
		align-items: start;
		justify-content: space-between;
		gap: 1rem;
		padding: 1.25rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 24px;
		background: linear-gradient(135deg, rgba(255, 255, 255, 0.82), rgba(244, 236, 227, 0.95));
	}

	.winner-image {
		flex: 0 0 160px;
	}

	.winner-label {
		margin: 0;
		color: var(--site-text-muted);
		font-size: 0.8rem;
		font-weight: 700;
		letter-spacing: 0.12em;
		text-transform: uppercase;
	}

	.winner-card h2 {
		margin: 0.35rem 0 0;
		font-size: clamp(1.6rem, 3vw, 2.2rem);
	}

	.winner-meta {
		display: flex;
		flex-wrap: wrap;
		gap: 0.8rem;
		margin: 0.55rem 0 0;
		color: var(--site-text-soft);
	}

	.processed-name {
		margin: 0.35rem 0 0;
		color: var(--site-text-muted);
		font-size: 0.82rem;
	}

	.winner-badges {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
	}

	.winner-badges span {
		display: inline-flex;
		align-items: center;
		padding: 0.55rem 0.8rem;
		border-radius: 999px;
		background: rgba(39, 31, 26, 0.08);
		color: var(--site-text-soft);
		font-size: 0.78rem;
		font-weight: 700;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.result-list,
	.per-image-stack {
		display: grid;
		gap: 0.95rem;
		margin-top: 1.2rem;
	}

	.topk-toggle-row {
		margin-top: 1rem;
	}

	.per-image-card {
		padding-top: 0.2rem;
	}

	.compact-list {
		margin-top: 0.9rem;
	}

	.result-card {
		padding: 1rem 1rem 1.1rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 20px;
		background: rgba(255, 255, 255, 0.76);
	}

	.result-head {
		display: flex;
		align-items: start;
		justify-content: space-between;
		gap: 0.8rem;
		margin-bottom: 0.8rem;
	}

	.result-rank {
		margin: 0;
		color: var(--site-text-muted);
		font-size: 0.75rem;
		font-weight: 700;
		letter-spacing: 0.12em;
		text-transform: uppercase;
	}

	.result-head h3 {
		margin: 0.25rem 0 0;
	}

	.result-actions {
		margin-top: 0.75rem;
	}

	.button.compact {
		padding: 0.65rem 0.9rem;
		font-size: 0.9rem;
	}

	.preview-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
		gap: 0.7rem;
		margin-top: 0.9rem;
	}

	.preview-grid img {
		display: block;
		width: 100%;
		height: 120px;
		object-fit: cover;
		border-radius: 14px;
		border: 1px solid rgba(77, 58, 46, 0.1);
	}

	.empty-inline {
		margin: 0;
		color: var(--site-text-muted);
	}

	@media (max-width: 900px) {
		.analyze-grid {
			grid-template-columns: 1fr;
		}

		.winner-card {
			flex-direction: column;
		}

		.winner-image {
			width: 100%;
			flex-basis: auto;
		}
	}
</style>
