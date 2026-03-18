<script>
	import { onMount } from 'svelte';
	import favicon from '$lib/assets/favicon.svg';
	import Navbar from '$lib/components/Navbar.svelte';
	import { loadAuth } from '$lib/stores/auth';
	import '$lib/styles/app.css';

	let { children } = $props();

	onMount(() => {
		loadAuth();
	});
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<div class="app-shell">
	<Navbar />
	{@render children()}
</div>
