<script>
	import { goto } from '$app/navigation';
	import { apiBaseUrl } from '$lib/config/env';
	import { saveAuth } from '$lib/stores/auth';

	let username = '';
	let password = '';
	let loading = false;
	let error = '';

	async function login(event) {
		event.preventDefault();
		error = '';
		loading = true;
		try {
			const response = await fetch(`${apiBaseUrl}/auth/login`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ username, password })
			});
			const data = await response.json();
			if (!response.ok) {
				throw new Error(data?.detail || 'Login failed.');
			}

			saveAuth(data.token, data.user);
			goto('/');
		} catch (err) {
			error = err.message || 'Login failed.';
		} finally {
			loading = false;
		}
	}
</script>

<svelte:head>
	<title>Login | Sneaker Matcher</title>
</svelte:head>

<section class="page-grid">
	<section class="panel auth-panel">
		<p class="eyebrow">Login</p>
		<h1>Sign in</h1>
		<p class="field-help">Use one of the proof-of-concept accounts or create your own.</p>

		<div class="demo-box">
			<p><strong>User:</strong> username <code>user</code>, password <code>user</code></p>
			<p><strong>Admin:</strong> username <code>admin</code>, password <code>admin</code></p>
		</div>

		<form class="auth-form" onsubmit={login}>
			<div class="field-grid">
				<label class="field-label" for="username">
					<span>Username</span>
					<input id="username" bind:value={username} autocomplete="username" required />
				</label>

				<label class="field-label" for="password">
					<span>Password</span>
					<input
						id="password"
						type="password"
						bind:value={password}
						autocomplete="current-password"
						required
					/>
				</label>
			</div>

			{#if error}
				<p class="field-error">{error}</p>
			{/if}

			<button class="button primary" type="submit" disabled={loading}>
				{loading ? 'Signing in...' : 'Login'}
			</button>
		</form>
	</section>
</section>
