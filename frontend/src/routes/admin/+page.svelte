<script>
	import { onMount } from 'svelte';
	import AdminCatalogPanel from '$lib/components/AdminCatalogPanel.svelte';
	import { apiBaseUrl } from '$lib/config/env';
	import { auth } from '$lib/stores/auth';

	let loading = true;
	let error = '';
	let users = [];
	let activeAction = '';
	let currentToken = null;
	let currentUserId = null;
	let activeTab = 'users';

	function adminPageError(data, fallback) {
		if (data && data.detail) {
			return data.detail;
		}

		return fallback;
	}

	function updateUserRoleInList(userId, role) {
		const updatedUsers = [];

		for (const user of users) {
			if (user.id === userId) {
				updatedUsers.push({
					...user,
					role,
					is_current_user: user.is_current_user
				});
				continue;
			}

			updatedUsers.push(user);
		}

		users = updatedUsers;
	}

	async function loadUsers(token) {
		loading = true;
		error = '';
		try {
			const response = await fetch(`${apiBaseUrl}/admin/users`, {
				headers: {
					Authorization: `Bearer ${token}`
				}
			});
			const data = await response.json();
			if (!response.ok) {
				throw new Error(adminPageError(data, 'Failed to load users.'));
			}
			if (data.users) {
				users = data.users;
			} else {
				users = [];
			}
		} catch (err) {
			if (err && err.message) {
				error = err.message;
			} else {
				error = 'Failed to load users.';
			}
		} finally {
			loading = false;
		}
	}

	async function setRole(userId, role) {
		if (!currentToken) return;
		activeAction = `${userId}:${role}`;
		error = '';
		try {
			const response = await fetch(`${apiBaseUrl}/admin/users/${userId}`, {
				method: 'PATCH',
				headers: {
					'Content-Type': 'application/json',
					Authorization: `Bearer ${currentToken}`
				},
				body: JSON.stringify({ role })
			});
			const data = await response.json();
			if (!response.ok) {
				throw new Error(adminPageError(data, 'Failed to update role.'));
			}
			updateUserRoleInList(userId, data.role);
		} catch (err) {
			if (err && err.message) {
				error = err.message;
			} else {
				error = 'Failed to update role.';
			}
		} finally {
			activeAction = '';
		}
	}

	async function deleteUser(userId) {
		if (!currentToken) return;
		if (!confirm('Delete this user and all of their catalogs?')) return;
		activeAction = `${userId}:delete`;
		error = '';
		try {
			const response = await fetch(`${apiBaseUrl}/admin/users/${userId}`, {
				method: 'DELETE',
				headers: {
					Authorization: `Bearer ${currentToken}`
				}
			});
			const data = await response.json();
			if (!response.ok) {
				throw new Error(adminPageError(data, 'Failed to delete user.'));
			}
			users = users.filter((user) => user.id !== data.id);
		} catch (err) {
			if (err && err.message) {
				error = err.message;
			} else {
				error = 'Failed to delete user.';
			}
		} finally {
			activeAction = '';
		}
	}

	function adminCount() {
		return users.filter((user) => user.role === 'admin').length;
	}

	onMount(() => {
		const unsubscribe = auth.subscribe((state) => {
			currentToken = state.token;
			if (state.user && state.user.id) {
				currentUserId = state.user.id;
			} else {
				currentUserId = null;
			}

			if (!state.user) {
				loading = false;
				error = 'Log in with an admin account to manage users.';
				users = [];
				return;
			}

			if (state.user.role !== 'admin') {
				loading = false;
				error = 'Admin access required.';
				users = [];
				return;
			}

			loadUsers(state.token);
		});

		return unsubscribe;
	});
</script>

<svelte:head>
	<title>Admin | Sneaker Matcher</title>
</svelte:head>

<section class="page-grid">
	<section class="panel section-card">
		<p class="eyebrow">Admin</p>
		<h1>Administration</h1>

		{#if !loading && !error}
			<div class="admin-tabs" role="tablist" aria-label="Admin sections">
				<button
					type="button"
					role="tab"
					class:active={activeTab === 'users'}
					class="admin-tab"
					onclick={() => (activeTab = 'users')}
				>
					User Management
				</button>
				<button
					type="button"
					role="tab"
					class:active={activeTab === 'catalog'}
					class="admin-tab"
					onclick={() => (activeTab = 'catalog')}
				>
					Catalog Metadata
				</button>
			</div>
		{/if}

		{#if loading}
			<p class="route-note">Loading users...</p>
		{:else if error}
			<p class="route-note route-error">{error}</p>
		{:else if activeTab === 'users'}
			<p class="route-note">
				View accounts, promote or demote admins, and remove test users.
			</p>
			<div class="summary-row">
				<div class="summary-card">
					<span>Total users</span>
					<strong>{users.length}</strong>
				</div>
				<div class="summary-card">
					<span>Admins</span>
					<strong>{adminCount()}</strong>
				</div>
			</div>

			<div class="user-grid">
				{#each users as user}
					<article class="user-card">
						<div class="user-head">
							<div>
								<h3>{user.username}</h3>
								<p>{user.full_name || user.username}</p>
							</div>
							<span class:admin={user.role === 'admin'} class="role-badge">{user.role}</span>
						</div>

						<dl class="meta-list">
							<div>
								<dt>Email</dt>
								<dd>{user.email}</dd>
							</div>
							<div>
								<dt>Catalogs</dt>
								<dd>{user.catalog_count}</dd>
							</div>
							<div>
								<dt>Items</dt>
								<dd>{user.item_count}</dd>
							</div>
							<div>
								<dt>Created</dt>
								<dd>{new Date(user.created_at).toLocaleDateString()}</dd>
							</div>
						</dl>

						<div class="action-row">
							{#if user.role === 'admin'}
								<button
									class="button ghost"
									type="button"
									disabled={activeAction !== '' || user.id === currentUserId}
									onclick={() => setRole(user.id, 'user')}
								>
									{activeAction === `${user.id}:user` ? 'Updating...' : 'Make user'}
								</button>
							{:else}
								<button
									class="button primary"
									type="button"
									disabled={activeAction !== ''}
									onclick={() => setRole(user.id, 'admin')}
								>
									{activeAction === `${user.id}:admin` ? 'Updating...' : 'Make admin'}
								</button>
							{/if}

							<button
								class="button danger"
								type="button"
								disabled={activeAction !== '' || user.id === currentUserId}
								onclick={() => deleteUser(user.id)}
							>
								{activeAction === `${user.id}:delete` ? 'Deleting...' : 'Delete'}
							</button>
						</div>
					</article>
				{/each}
			</div>
		{:else if activeTab === 'catalog'}
			<AdminCatalogPanel token={currentToken} />
		{/if}
	</section>
</section>

<style>
	.route-note {
		margin: 0.6rem 0 0;
		color: var(--site-text-soft);
	}

	.route-error {
		color: var(--color-error);
	}

	.admin-tabs {
		display: flex;
		flex-wrap: wrap;
		gap: 0.75rem;
		margin-top: 1rem;
	}

	.admin-tab {
		padding: 0.85rem 1rem;
		border: 1px solid rgba(77, 58, 46, 0.14);
		border-radius: 12px;
		background: rgba(255, 252, 247, 0.84);
		color: var(--site-text-soft);
		font: inherit;
		font-weight: 700;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		cursor: pointer;
		transition: background 140ms ease, transform 140ms ease, box-shadow 140ms ease;
	}

	.admin-tab.active {
		background: var(--site-selected);
		color: var(--site-text);
		box-shadow: inset 0 0 0 1px rgba(212, 87, 46, 0.18);
	}

	.summary-row {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(180px, 220px));
		gap: 0.9rem;
		margin-top: 1.25rem;
	}

	.summary-card {
		padding: 1rem 1.1rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 18px;
		background: rgba(255, 255, 255, 0.72);
	}

	.summary-card span {
		display: block;
		color: var(--site-text-muted);
		font-size: 0.82rem;
		font-weight: 700;
		letter-spacing: 0.1em;
		text-transform: uppercase;
	}

	.summary-card strong {
		display: block;
		margin-top: 0.3rem;
		font-size: 2rem;
	}

	.user-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
		gap: 1rem;
		margin-top: 1.25rem;
	}

	.user-card {
		padding: 1.2rem;
		border: 1px solid rgba(77, 58, 46, 0.12);
		border-radius: 22px;
		background: rgba(255, 255, 255, 0.76);
	}

	.user-head {
		display: flex;
		align-items: start;
		justify-content: space-between;
		gap: 0.8rem;
	}

	.user-head h3 {
		margin: 0;
	}

	.user-head p {
		margin: 0.25rem 0 0;
		color: var(--site-text-muted);
	}

	.role-badge {
		display: inline-flex;
		align-items: center;
		padding: 0.45rem 0.7rem;
		border-radius: 999px;
		background: rgba(39, 31, 26, 0.08);
		color: var(--site-text-soft);
		font-size: 0.78rem;
		font-weight: 700;
		letter-spacing: 0.1em;
		text-transform: uppercase;
	}

	.role-badge.admin {
		background: rgba(212, 87, 46, 0.12);
		color: #b45534;
	}

	.meta-list {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.8rem;
		margin: 1rem 0 0;
	}

	.meta-list dt {
		color: var(--site-text-muted);
		font-size: 0.74rem;
		font-weight: 700;
		letter-spacing: 0.1em;
		text-transform: uppercase;
	}

	.meta-list dd {
		margin: 0.25rem 0 0;
		color: var(--site-text);
	}

	.action-row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.7rem;
		margin-top: 1.1rem;
	}

	.button.danger {
		background: rgba(161, 63, 32, 0.12);
		color: var(--color-error);
	}

	.button:disabled {
		opacity: 0.55;
		cursor: not-allowed;
		transform: none;
	}
</style>
