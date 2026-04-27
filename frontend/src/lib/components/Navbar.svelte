<script>
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { authNav, primaryNav } from '$lib/config/navigation';
	import { auth, clearAuth } from '$lib/stores/auth';

	function isActive(href) {
		return href === '/' ? page.url.pathname === '/' : page.url.pathname.startsWith(href);
	}

	function logout() {
		clearAuth();
		goto('/');
	}
</script>

<header class="site-nav">
	<a class="brand" href="/">
		<!-- Logo adapted from https://similarpng.com/red-sneakers-outfit-men-on-transparent-background-png/ -->
		<img class="brand-logo" src="/sneaker-logo.png" alt="Sneaker Matcher logo" />
		<div class="brand-copy">
			<strong>Sneaker Matcher</strong>
			<span>Visual matching workspace</span>
		</div>
	</a>

	<nav class="nav-links" aria-label="Primary">
		{#each primaryNav as item}
			<a class:active={isActive(item.href)} href={item.href}>{item.label}</a>
		{/each}
		{#if $auth.user?.role === 'admin'}
			<a class:active={isActive('/admin')} href="/admin">Admin</a>
		{/if}
	</nav>

	<div class="auth-links">
		{#if $auth.user}
			<div class="auth-pill">
				<strong>{$auth.user.username}</strong>
				<span>{$auth.user.role}</span>
			</div>
			<button class="auth-link" type="button" onclick={logout}>Logout</button>
		{:else}
			{#each authNav as item}
				<a class:signup={item.label === 'Sign Up'} class="auth-link" href={item.href}>
					{item.label}
				</a>
			{/each}
		{/if}
	</div>
</header>
