<script>
	import { onMount } from 'svelte';
	import Navbar from '$lib/components/Navbar.svelte';
	import { loadAuth } from '$lib/stores/auth';
	import '$lib/styles/app.css';

	let { children } = $props();

	onMount(() => {
		loadAuth();
	});
</script>

<svelte:head>
	<!-- Logo adapted from https://similarpng.com/red-sneakers-outfit-men-on-transparent-background-png/ -->
	<link rel="icon" href="/sneaker-logo.png" />
</svelte:head>

<div class="app-shell">
	<Navbar />
	{@render children()}
</div>
