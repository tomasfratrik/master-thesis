<script>
	import { apiBaseUrl } from '$lib/config/env';

	const aggregationOptions = [
		{ value: 'logit_mean', label: 'Logit mean', note: 'Best default for one sneaker with multiple views.' },
		{ value: 'embedding_mean', label: 'Embedding mean', note: 'Average image embeddings before classification.' },
		{ value: 'prob_mean', label: 'Probability mean', note: 'Average per-image class probabilities.' }
	];
	const analysisMethodOptions = [
		{
			value: 'classifier',
			label: 'Classifier',
			note: 'Existing prompt-based fine-grained label prediction.'
		},
		{
			value: 'retrieval',
			label: 'Embeddings',
			note: 'Compare the query against stored sneaker catalog embeddings.'
		}
	];
	const signalTags = ['visual search', 'embedding retrieval', 'prototype classes', 'clip', 'ranking'];

	let analysisMethod = 'classifier';
	let mode = 'grouped';
	let aggregation = 'logit_mean';
	let selectedFiles = [];
	let filePreviews = [];
	let loading = false;
	let error = '';
	let response = null;
	let expandedPreviewKeys = new Set();
	let expandedTopKKeys = new Set();
	let expandedMetadataKeys = new Set();
	let lightbox = null;

	function fileId(file, index) {
		return `${file.name}-${file.size}-${file.lastModified}-${index}`;
	}

	function syncPreviews(files) {
		for (const preview of filePreviews) {
			URL.revokeObjectURL(preview.url);
		}
		filePreviews = files.map((file, index) => ({
			id: fileId(file, index),
			name: file.name,
			url: URL.createObjectURL(file)
		}));
	}

	function mergeFiles(currentFiles, nextFiles) {
		const merged = [...currentFiles];
		for (const file of nextFiles) {
			const exists = merged.some(
				(current) =>
					current.name === file.name &&
					current.size === file.size &&
					current.lastModified === file.lastModified
			);
			if (!exists) {
				merged.push(file);
			}
		}
		return merged;
	}

	function updateFiles(list, append = false) {
		let nextFiles = [];
		if (list) {
			nextFiles = Array.from(list);
		}

		if (append) {
			selectedFiles = mergeFiles(selectedFiles, nextFiles);
		} else {
			selectedFiles = nextFiles;
		}
		syncPreviews(selectedFiles);
		response = null;
		error = '';
		expandedPreviewKeys = new Set();
		expandedTopKKeys = new Set();
		expandedMetadataKeys = new Set();
	}

	function removeFile(index) {
		selectedFiles = selectedFiles.filter((_, currentIndex) => currentIndex !== index);
		syncPreviews(selectedFiles);
		response = null;
		error = '';
		expandedPreviewKeys = new Set();
		expandedTopKKeys = new Set();
		expandedMetadataKeys = new Set();
	}

	function toggleMode(nextMode) {
		mode = nextMode;
		response = null;
		error = '';
		expandedPreviewKeys = new Set();
		expandedTopKKeys = new Set();
		expandedMetadataKeys = new Set();
	}

	function prettyPercent(score) {
		return `${(score * 100).toFixed(1)}%`;
	}

	function formatEuroPrice(value) {
		const numeric = Number(value);
		if (!Number.isFinite(numeric)) {
			return null;
		}

		return `EUR ${numeric.toFixed(2)}`;
	}

	function candidateMetadataLines(candidate) {
		const product = candidate?.product;
		if (!product) {
			return [];
		}

		const lines = [];
		if (product.brand) {
			lines.push(`Brand: ${product.brand}`);
		}
		if (product.model) {
			lines.push(`Model: ${product.model}`);
		}
		if (product.colorway) {
			lines.push(`Colorway: ${product.colorway}`);
		}
		if (product.sku) {
			lines.push(`SKU: ${product.sku}`);
		}
		if (product.release_year) {
			lines.push(`Release year: ${product.release_year}`);
		}
		if (product.price_eur !== undefined && product.price_eur !== null && product.price_eur !== '') {
			const formatted = formatEuroPrice(product.price_eur);
			if (formatted) {
				lines.push(`Price: ${formatted}`);
			}
		}
		return lines;
	}

	function candidateExtraMetadataLines(candidate) {
		const product = candidate?.product;
		if (!product) {
			return [];
		}

		const lines = [];
		if (product.retail_price !== undefined && product.retail_price !== null && product.retail_price !== '') {
			lines.push(`Retail price: ${product.retail_price}${product.currency ? ` ${product.currency}` : ''}`);
		}
		if (product.release_date) {
			lines.push(`Release date: ${product.release_date}`);
		}
		if (product.gender) {
			lines.push(`Gender: ${product.gender}`);
		}
		if (product.category) {
			lines.push(`Category: ${product.category}`);
		}
		if (product.source) {
			lines.push(`Source: ${product.source}`);
		}
		if (product.source_url) {
			lines.push(`Source URL: ${product.source_url}`);
		}
		if (product.description) {
			lines.push(`Description: ${product.description}`);
		}
		return lines;
	}

	function previewUrl(url) {
		if (!url) {
			return url;
		}

		if (url.startsWith('http')) {
			return url;
		}

		return `${apiBaseUrl}${url}`;
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

	function analysisMethodLabel() {
		if (analysisMethod === 'retrieval') {
			return 'Embeddings';
		}

		return 'Classifier';
	}

	function modeLabel() {
		if (mode === 'grouped') {
			return 'One Sneaker';
		}

		return 'Multiple Sneakers';
	}

	function loadingMessage() {
		if (analysisMethod === 'retrieval') {
			return 'Preprocessing images and comparing against catalog embeddings.';
		}

		return 'Preprocessing images and running prediction.';
	}

	function analyzeButtonLabel() {
		if (loading) {
			return 'Analyzing...';
		}

		return 'Analyze Photos';
	}

	function resultMethodLabel() {
		if (response?.analysis_method === 'retrieval') {
			return 'embeddings';
		}

		return 'classifier';
	}

	function topKToggleLabel(key) {
		if (expandedTopKKeys.has(key)) {
			return 'Hide top 5 results';
		}

		return 'Show top 5 results';
	}

	function previewToggleLabel(key) {
		if (expandedPreviewKeys.has(key)) {
			return 'Hide previews';
		}

		return 'Show previews';
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

	function toggleMetadata(key) {
		const next = new Set(expandedMetadataKeys);
		if (next.has(key)) {
			next.delete(key);
		} else {
			next.add(key);
		}
		expandedMetadataKeys = next;
	}

	function openLightbox(src, label) {
		lightbox = { src, label };
	}

	function closeLightbox() {
		lightbox = null;
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
		expandedMetadataKeys = new Set();

		try {
			const formData = new FormData();
			for (const file of selectedFiles) {
				formData.append('files', file);
			}

			const params = new URLSearchParams({
				mode,
				top_k: '5'
			});
			if (mode === 'grouped' && analysisMethod === 'classifier') {
				params.set('aggregation', aggregation);
			}

			let endpoint = '/analyze';
			if (analysisMethod === 'retrieval') {
				endpoint = '/analyze-retrieval';
			}
			const res = await fetch(`${apiBaseUrl}${endpoint}?${params.toString()}`, {
				method: 'POST',
				body: formData
			});
			const data = await res.json();
			if (!res.ok) {
				if (data && data.detail) {
					throw new Error(data.detail);
				}

				throw new Error('Analysis failed.');
			}
			response = data;
		} catch (err) {
			if (err && err.message) {
				error = err.message;
			} else {
				error = 'Analysis failed.';
			}
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
			<div class="aggregation-block">
				<p class="eyebrow">Matching Method</p>
				<div class="aggregation-grid">
					{#each analysisMethodOptions as option}
						<button
							type="button"
							class:active={analysisMethod === option.value}
							class="aggregation-card"
							onclick={() => (analysisMethod = option.value)}
						>
							<strong>{option.label}</strong>
							<span>{option.note}</span>
						</button>
					{/each}
				</div>
			</div>

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
					onchange={(event) => {
						updateFiles(event.currentTarget.files, true);
						event.currentTarget.value = '';
					}}
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
					{#each filePreviews as file, index}
						<figure class="upload-preview-card">
							<button class="image-action remove" type="button" onclick={() => removeFile(index)}>
								×
							</button>
							<button class="image-action expand" type="button" onclick={() => openLightbox(file.url, file.name)}>
								Expand
							</button>
							<div class="image-frame upload-frame">
								<img src={file.url} alt={file.name} />
							</div>
							<figcaption>{file.name}</figcaption>
						</figure>
					{/each}
				</div>
			{/if}

			{#if mode === 'grouped' && analysisMethod === 'classifier'}
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
					{analyzeButtonLabel()}
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
			<p class="eyebrow">Run Summary</p>
			<div class="summary-stack">
				<div class="summary-line">
					<span>Method</span>
					<strong>{analysisMethodLabel()}</strong>
				</div>
				<div class="summary-line">
					<span>Mode</span>
					<strong>{modeLabel()}</strong>
				</div>
				<div class="summary-line">
					<span>Files selected</span>
					<strong>{selectedFiles.length}</strong>
				</div>
				{#if mode === 'grouped' && analysisMethod === 'classifier'}
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
					<p>{loadingMessage()}</p>
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
									<button
										class="image-action expand"
										type="button"
										onclick={() => openLightbox(image.data_url, image.filename)}
									>
										Expand
									</button>
									<div class="image-frame analyzed-frame">
										<img src={image.data_url} alt={image.filename} />
									</div>
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
						{#if candidateMetadataLines(result).length}
							<div class="candidate-meta">
								{#each candidateMetadataLines(result) as line}
									<span>{line}</span>
								{/each}
							</div>
						{/if}
						{#if candidateExtraMetadataLines(result).length}
							<div class="result-actions metadata-actions">
								<button class="button ghost compact" type="button" onclick={() => toggleMetadata('grouped-winner')}>
									{expandedMetadataKeys.has('grouped-winner') ? 'Hide metadata' : 'Show metadata'}
								</button>
							</div>
							{#if expandedMetadataKeys.has('grouped-winner')}
								<div class="candidate-meta extra-meta">
									{#each candidateExtraMetadataLines(result) as line}
										<span>{line}</span>
									{/each}
								</div>
							{/if}
						{/if}
					</div>
					<div class="winner-badges">
						<span>{response.query_image_count} photos</span>
						<span>{response.processed_image_count} crops</span>
						<span>{result.aggregation}</span>
						<span>{resultMethodLabel()}</span>
					</div>
				</div>

				<div class="topk-toggle-row">
					<button class="button ghost compact" type="button" onclick={() => toggleTopK('grouped-topk')}>
						{topKToggleLabel('grouped-topk')}
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
								{#if candidateMetadataLines(candidate).length}
									<div class="candidate-meta">
										{#each candidateMetadataLines(candidate) as line}
											<span>{line}</span>
										{/each}
									</div>
								{/if}
								<progress class="progress progress-info w-full" value={candidate.score * 100} max="100"></progress>
								<div class="result-actions">
									{#if candidateExtraMetadataLines(candidate).length}
										<button class="button ghost compact" type="button" onclick={() => toggleMetadata(`grouped-meta-${index}`)}>
											{expandedMetadataKeys.has(`grouped-meta-${index}`) ? 'Hide metadata' : 'Show metadata'}
										</button>
									{/if}
									<button class="button ghost compact" type="button" onclick={() => togglePreview(key)}>
										{previewToggleLabel(key)}
									</button>
								</div>
								{#if candidateExtraMetadataLines(candidate).length && expandedMetadataKeys.has(`grouped-meta-${index}`)}
									<div class="candidate-meta extra-meta">
										{#each candidateExtraMetadataLines(candidate) as line}
											<span>{line}</span>
										{/each}
									</div>
								{/if}
								{#if expandedPreviewKeys.has(key)}
									<div class="preview-grid">
										{#if candidate.preview_urls?.length}
											{#each candidate.preview_urls as url}
												<div class="preview-tile">
													<button
														class="image-action expand"
														type="button"
														onclick={() => openLightbox(previewUrl(url), candidate.label)}
													>
														Expand
													</button>
													<div class="image-frame preview-frame">
														<img src={previewUrl(url)} alt={candidate.label} />
													</div>
												</div>
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
										<button
											class="image-action expand"
											type="button"
											onclick={() => openLightbox(result.processed_image.data_url, result.query_filename)}
										>
											Expand
										</button>
										<div class="image-frame winner-frame">
											<img src={result.processed_image.data_url} alt={result.query_filename} />
										</div>
									</div>
								{/if}
								<div>
									<p class="winner-label">{result.query_filename}</p>
									<p class="processed-name">{result.processed_filename}</p>
									<h2>{result.label}</h2>
									<p class="winner-meta">Score {prettyPercent(result.score)}</p>
									{#if candidateMetadataLines(result).length}
										<div class="candidate-meta">
											{#each candidateMetadataLines(result) as line}
												<span>{line}</span>
											{/each}
										</div>
									{/if}
									{#if candidateExtraMetadataLines(result).length}
										<div class="result-actions metadata-actions">
											<button class="button ghost compact" type="button" onclick={() => toggleMetadata(`per-winner-${resultIndex}`)}>
												{expandedMetadataKeys.has(`per-winner-${resultIndex}`) ? 'Hide metadata' : 'Show metadata'}
											</button>
										</div>
										{#if expandedMetadataKeys.has(`per-winner-${resultIndex}`)}
											<div class="candidate-meta extra-meta">
												{#each candidateExtraMetadataLines(result) as line}
													<span>{line}</span>
												{/each}
											</div>
										{/if}
									{/if}
								</div>
								<div class="winner-badges">
									<span>{result.prepared_source}</span>
									<span>{resultMethodLabel()}</span>
								</div>
							</div>

							<div class="topk-toggle-row">
								<button
									class="button ghost compact"
									type="button"
									onclick={() => toggleTopK(`per-topk-${resultIndex}`)}
								>
									{topKToggleLabel(`per-topk-${resultIndex}`)}
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
											{#if candidateMetadataLines(candidate).length}
												<div class="candidate-meta">
													{#each candidateMetadataLines(candidate) as line}
														<span>{line}</span>
													{/each}
												</div>
											{/if}
											<progress class="progress progress-info w-full" value={candidate.score * 100} max="100"></progress>
											<div class="result-actions">
												{#if candidateExtraMetadataLines(candidate).length}
													<button class="button ghost compact" type="button" onclick={() => toggleMetadata(`per-meta-${resultIndex}-${index}`)}>
														{expandedMetadataKeys.has(`per-meta-${resultIndex}-${index}`) ? 'Hide metadata' : 'Show metadata'}
													</button>
												{/if}
												<button class="button ghost compact" type="button" onclick={() => togglePreview(key)}>
													{previewToggleLabel(key)}
												</button>
											</div>
											{#if candidateExtraMetadataLines(candidate).length && expandedMetadataKeys.has(`per-meta-${resultIndex}-${index}`)}
												<div class="candidate-meta extra-meta">
													{#each candidateExtraMetadataLines(candidate) as line}
														<span>{line}</span>
													{/each}
												</div>
											{/if}
											{#if expandedPreviewKeys.has(key)}
												<div class="preview-grid">
													{#if candidate.preview_urls?.length}
														{#each candidate.preview_urls as url}
															<div class="preview-tile">
																<button
																	class="image-action expand"
																	type="button"
																	onclick={() => openLightbox(previewUrl(url), candidate.label)}
																>
																	Expand
																</button>
																<div class="image-frame preview-frame">
																	<img src={previewUrl(url)} alt={candidate.label} />
																</div>
															</div>
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

	{#if lightbox}
		<div class="lightbox-backdrop" role="button" tabindex="0" onclick={closeLightbox} onkeydown={(event) => event.key === 'Escape' && closeLightbox()}>
			<div
				class="lightbox-panel"
				role="dialog"
				aria-modal="true"
				tabindex="-1"
				onclick={(event) => event.stopPropagation()}
				onkeydown={(event) => event.stopPropagation()}
			>
				<button class="image-action remove lightbox-close" type="button" onclick={closeLightbox}>×</button>
				<div class="lightbox-frame">
					<img src={lightbox.src} alt={lightbox.label} />
				</div>
				<p>{lightbox.label}</p>
			</div>
		</div>
	{/if}
</section>

<style>
	.analyze-grid {
		grid-template-columns: minmax(0, 1.3fr) minmax(280px, 360px);
		align-items: start;
	}

		.display-title span {
			display: block;
			background: linear-gradient(135deg, #d7885f, #e7b76a);
			-webkit-background-clip: text;
			background-clip: text;
			color: transparent;
		}

		.signal-row {
			display: flex;
			flex-wrap: wrap;
			gap: 0.55rem;
			margin-top: 1rem;
			padding: 0.95rem 0;
			border-top: 1px solid rgba(87, 67, 50, 0.14);
			border-bottom: 1px solid rgba(87, 67, 50, 0.14);
		}

		.signal-row span {
			padding: 0.55rem 0.7rem;
			border: 1px solid rgba(87, 67, 50, 0.14);
			border-radius: 999px;
			background: rgba(255, 252, 247, 0.82);
			color: #8a7563;
			font-size: 0.76rem;
			font-weight: 800;
			letter-spacing: 0.14em;
			text-transform: uppercase;
	}

	.mode-tabs {
		display: flex;
		gap: 0.75rem;
		margin-top: 1.3rem;
	}

		.mode-tab {
			padding: 0.85rem 1rem;
			border: 1px solid rgba(77, 58, 46, 0.14);
			border-radius: 10px;
			background: rgba(255, 252, 247, 0.84);
			color: var(--site-text-soft);
			font: inherit;
			font-weight: 700;
			letter-spacing: 0.08em;
		text-transform: uppercase;
		cursor: pointer;
		transition: background 140ms ease, transform 140ms ease, box-shadow 140ms ease;
	}

		.mode-tab.active {
			background: var(--site-selected);
			color: var(--site-text);
			box-shadow: inset 0 0 0 1px rgba(212, 87, 46, 0.18);
		}

	.mode-note {
		margin: 0.9rem 0 0;
		color: var(--site-text-muted);
	}

		.upload-zone {
			display: block;
			margin-top: 1.25rem;
			padding: 1.25rem;
			border: 1px dashed rgba(212, 87, 46, 0.28);
			border-radius: 12px;
			background:
				linear-gradient(135deg, rgba(255, 252, 247, 0.92), rgba(242, 231, 218, 0.92)),
				repeating-linear-gradient(
					-45deg,
					rgba(212, 87, 46, 0.03) 0,
					rgba(212, 87, 46, 0.03) 12px,
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
		font-size: 1.05rem;
		letter-spacing: -0.03em;
	}

	.upload-zone span {
		display: block;
		margin-top: 0.45rem;
		color: var(--site-text-soft);
		line-height: 1.5;
	}

	.upload-preview-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 0.8rem;
		margin-top: 1rem;
	}

		.upload-preview-card {
			position: relative;
			margin: 0;
			padding: 0.65rem;
			width: 180px;
			border: 1px solid rgba(77, 58, 46, 0.12);
			border-radius: 10px;
			background: rgba(255, 252, 247, 0.82);
		}

		.image-frame {
			display: grid;
			place-items: center;
			overflow: hidden;
			border-radius: 8px;
			background: linear-gradient(135deg, rgba(245, 239, 231, 0.96), rgba(232, 220, 205, 0.96));
		}

	.upload-frame,
	.analyzed-frame,
	.preview-frame,
	.winner-frame {
		width: 100%;
		height: 150px;
	}

	.image-frame img {
		display: block;
		width: 100%;
		height: 100%;
		object-fit: contain;
	}

	.upload-preview-card figcaption {
		margin-top: 0.55rem;
		color: var(--site-text-muted);
		font-size: 0.8rem;
		word-break: break-word;
	}

	.image-action {
		position: absolute;
		z-index: 1;
		padding: 0.4rem 0.65rem;
		border: none;
		border-radius: 999px;
			background: rgba(39, 31, 26, 0.88);
			color: #fff4ec;
		font: inherit;
		font-size: 0.75rem;
		font-weight: 700;
		cursor: pointer;
	}

	.image-action.expand {
		top: 0.95rem;
		right: 0.9rem;
	}

	.image-action.remove {
		top: 0.95rem;
		left: 0.9rem;
		width: 2rem;
		height: 2rem;
		padding: 0;
		font-size: 1.1rem;
		line-height: 1;
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
			border: 1px solid rgba(77, 58, 46, 0.14);
			border-radius: 10px;
			background: rgba(255, 252, 247, 0.84);
			text-align: left;
			cursor: pointer;
		}

		.aggregation-card.active {
			background: var(--site-selected);
			box-shadow: inset 0 0 0 1px rgba(212, 87, 46, 0.16);
		}

	.aggregation-card span {
		color: var(--site-text-muted);
		font-size: 0.88rem;
		line-height: 1.45;
	}

		.status-panel {
			padding: 1.8rem;
			background:
				linear-gradient(180deg, rgba(255, 252, 247, 0.96), rgba(247, 242, 234, 0.96)),
				linear-gradient(135deg, rgba(212, 87, 46, 0.05), transparent 38%);
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
			border-bottom: 1px solid rgba(87, 67, 50, 0.12);
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
			border: 1px solid rgba(77, 58, 46, 0.12);
			border-radius: 10px;
			background: rgba(255, 252, 247, 0.82);
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
		display: flex;
		flex-wrap: wrap;
		gap: 0.8rem;
		margin-top: 0.75rem;
	}

		.analyzed-card {
			position: relative;
			margin: 0;
			padding: 0.7rem;
			width: 220px;
			border: 1px solid rgba(77, 58, 46, 0.12);
			border-radius: 10px;
			background: rgba(255, 252, 247, 0.82);
		}


		.winner-card {
			display: flex;
			align-items: start;
			justify-content: space-between;
			gap: 1rem;
			padding: 1.25rem;
			border: 1px solid rgba(77, 58, 46, 0.14);
			border-radius: 12px;
			background:
				linear-gradient(135deg, rgba(255, 252, 247, 0.96), rgba(243, 235, 225, 0.96)),
				linear-gradient(90deg, transparent, rgba(212, 87, 46, 0.06), transparent);
		}

	.winner-image {
		position: relative;
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

	.candidate-meta {
		display: flex;
		flex-wrap: wrap;
		gap: 0.45rem;
		margin: 0.45rem 0 0.2rem;
	}

	.candidate-meta span {
		padding: 0.32rem 0.55rem;
		border: 1px solid rgba(77, 58, 46, 0.1);
		border-radius: 999px;
		background: rgba(255, 255, 255, 0.68);
		color: var(--site-text-muted);
		font-size: 0.78rem;
		font-weight: 700;
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
			border: 1px solid rgba(77, 58, 46, 0.12);
			border-radius: 999px;
			background: rgba(255, 248, 240, 0.84);
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
			border-radius: 10px;
			background: rgba(255, 252, 247, 0.82);
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
		display: flex;
		flex-wrap: wrap;
		gap: 0.65rem;
		align-items: center;
		margin-top: 0.75rem;
	}

	.extra-meta {
		margin-top: 0.6rem;
	}

	.button.compact {
		padding: 0.65rem 0.9rem;
		font-size: 0.9rem;
	}

	.preview-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 0.7rem;
		margin-top: 0.9rem;
	}

	.preview-tile {
		position: relative;
		width: 150px;
	}

		.preview-frame {
			height: 120px;
			border: 1px solid rgba(77, 58, 46, 0.12);
		}

	.empty-inline {
		margin: 0;
		color: var(--site-text-muted);
	}

	.lightbox-backdrop {
		position: fixed;
		inset: 0;
		z-index: 30;
		display: grid;
		place-items: center;
		padding: 1.25rem;
		background: rgba(4, 10, 16, 0.82);
	}

		.lightbox-panel {
			position: relative;
			width: min(90vw, 900px);
			padding: 1rem;
			border-radius: 12px;
			background: rgba(248, 242, 234, 0.98);
		}

		.lightbox-frame {
			display: grid;
			place-items: center;
			max-height: 75vh;
			overflow: hidden;
			border-radius: 8px;
			background: linear-gradient(135deg, rgba(245, 239, 231, 0.96), rgba(232, 220, 205, 0.96));
		}

	.lightbox-frame img {
		display: block;
		max-width: 100%;
		max-height: 75vh;
		object-fit: contain;
	}

	.lightbox-panel p {
		margin: 0.75rem 0 0;
		color: var(--site-text-soft);
	}

	.lightbox-close {
		top: 1rem;
		right: 1rem;
		left: auto;
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
