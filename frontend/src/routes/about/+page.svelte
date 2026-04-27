<svelte:head>
	<title>About | Sneaker Matcher</title>
</svelte:head>

<section class="page-grid">
	<section class="panel section-card about-card">
		<p class="eyebrow">About</p>
		<h1>About the project</h1>
		<p class="lead">
			Sneaker Matcher is a master thesis project focused on fine-grained sneaker recognition.
			The application analyzes uploaded sneaker images and returns the most likely matching
			sneaker classes together with ranked alternatives.
		</p>

		<div class="about-grid">
			<article class="about-block">
				<h2>What the application does</h2>
				<p>
					The system preprocesses uploaded images, extracts the relevant sneaker crop, and
					runs prediction with a fine-tuned vision-language model. The result view shows the
					top match, score, ranked alternatives, preview images, and available catalog
					metadata.
				</p>
			</article>

			<article class="about-block">
				<h2>Supported Sneakers</h2>
				<p>
					The <a href="/supported-sneakers">Supported Sneakers</a> page shows the sneaker
					classes currently supported by the loaded model checkpoint. In other words, it
					represents the active prediction space used by the application at runtime.
				</p>
			</article>

			<article class="about-block">
				<h2>Analysis Modes</h2>
				<p>
					One Sneaker mode combines multiple uploaded images as different views of the same
					sneaker. Multiple Sneakers mode analyzes each image separately and returns one
					result set per processed image or crop.
				</p>
			</article>

			<article class="about-block">
				<h2>Source Code</h2>
				<p>
					The implementation of this thesis project is available at
					<a href="https://github.com/tomasfratrik/master-thesis" target="_blank" rel="noreferrer">
						github.com/tomasfratrik/master-thesis
					</a>.
				</p>
			</article>
		</div>
	</section>
</section>

<style>
	.about-card {
		display: grid;
		gap: 1.2rem;
	}

	.lead {
		max-width: 70ch;
		margin: 0;
		color: var(--site-text-soft);
		line-height: 1.65;
	}

	.about-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
		gap: 1rem;
	}

	.about-block {
		padding: 1rem 1.05rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 18px;
		background: rgba(255, 252, 247, 0.72);
	}

	.about-block h2 {
		margin: 0 0 0.55rem;
		font-size: 1.05rem;
	}

	.about-block p {
		margin: 0;
		color: var(--site-text-soft);
		line-height: 1.6;
	}

	.about-block a {
		color: var(--site-accent, #b8542e);
		text-decoration: none;
	}

	.about-block a:hover {
		text-decoration: underline;
	}
</style>
