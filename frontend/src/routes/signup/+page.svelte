<script>
	import { goto } from '$app/navigation';
	import { apiBaseUrl } from '$lib/config/env';
	import { saveAuth } from '$lib/stores/auth';

	let username = '';
	let password = '';
	let confirmPassword = '';
	let loading = false;
	let error = '';

	async function signup(event) {
		event.preventDefault();
		error = '';
		if (password !== confirmPassword) {
			error = 'Passwords do not match.';
			return;
		}
		loading = true;
		try {
			const response = await fetch(`${apiBaseUrl}/auth/register`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					username,
					password
				})
			});
			const data = await response.json();
			if (!response.ok) {
				throw new Error(data?.detail || 'Sign up failed.');
			}

			saveAuth(data.token, data.user);
			goto('/');
		} catch (err) {
			error = err.message || 'Sign up failed.';
		} finally {
			loading = false;
		}
	}
</script>

<svelte:head>
	<title>Sign Up | Sneaker Matcher</title>
</svelte:head>

<section class="page-grid">
	<section class="panel auth-panel">
		<p class="eyebrow">Sign Up</p>
		<h1>Create account</h1>
		<p class="field-help">Only username and password are required.</p>

		<form class="auth-form" onsubmit={signup}>
			<div class="field-grid">
				<label class="field-label" for="signup-username">
					<span>Username</span>
					<input id="signup-username" bind:value={username} autocomplete="username" required />
				</label>

				<label class="field-label" for="signup-password">
					<span>Password</span>
					<input
						id="signup-password"
						type="password"
						bind:value={password}
						autocomplete="new-password"
						required
					/>
				</label>

				<label class="field-label" for="signup-password-confirm">
					<span>Confirm password</span>
					<input
						id="signup-password-confirm"
						type="password"
						bind:value={confirmPassword}
						autocomplete="new-password"
						required
					/>
				</label>
			</div>

			{#if error}
				<p class="field-error">{error}</p>
			{/if}

			<button class="button primary" type="submit" disabled={loading}>
				{loading ? 'Creating account...' : 'Sign Up'}
			</button>
		</form>
	</section>
</section>
