<svelte:head>
	<title>Sneaker Matcher</title>
	<meta
		name="description"
		content="Baseline frontend for Sneaker Matcher with a custom-styled layout on top of Tailwind."
	/>
</svelte:head>

<section class="page-grid">
	<div class="hero-grid">
		<section class="panel hero-copy">
			<p class="eyebrow">Sneaker Matching Platform</p>
			<h1 class="display-title">Catalog sneakers visually.</h1>
			<p class="lede">
				A clean baseline for Sneaker Matcher: structured navigation, shared design tokens, and
				reusable sections you can extend into upload, matching, and catalog management flows.
			</p>

			<div class="cta-row">
				<a class="button primary" href="/signup">Get Started</a>
				<a class="button ghost" href="/about">Learn More</a>
			</div>
		</section>

		<section class="panel section-card">
			<p class="eyebrow">Upload Pattern</p>
			<div class="upload-sample">
				<strong>Upload sneaker photos</strong>
				<p>
					This is just a visual sample block for now. Later you can replace it with the real
					drag-and-drop or multi-image upload component.
				</p>
			</div>
		</section>
	</div>

	<section class="card-grid">
		<article class="info-card">
			<h3>Catalogs</h3>
			<p>Create personal collections, upload references, and organize matched sneakers.</p>
		</article>
		<article class="info-card">
			<h3>Visual Search</h3>
			<p>Compare uploaded shoes against supported catalog entries and inspect top-ranked results.</p>
		</article>
		<article class="info-card">
			<h3>Metadata</h3>
			<p>Assign brand, model, and product details after the catalog match is confirmed.</p>
		</article>
	</section>

	<section class="split-grid">
		<section class="panel section-card">
			<p class="eyebrow">What This Branch Gives You</p>
			<ul class="muted-list">
				<li>Global app styling through a shared stylesheet.</li>
				<li>A reusable top navbar component.</li>
				<li>Separate route files instead of a single giant page.</li>
				<li>Tailwind and DaisyUI still available when you need primitives later.</li>
			</ul>
		</section>

		<section class="panel section-card">
			<p class="eyebrow">Next Good Steps</p>
			<ul class="muted-list">
				<li>Add a shared API client with `PUBLIC_API_BASE_URL`.</li>
				<li>Add auth pages behind the navbar buttons.</li>
				<li>Build the actual analysis and catalog flows on top of this baseline.</li>
				<li>Introduce alerts, loaders, and skeletons as your own reusable wrappers.</li>
			</ul>
		</section>
	</section>
</section>
