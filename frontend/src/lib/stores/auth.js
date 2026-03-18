import { browser } from '$app/environment';
import { writable } from 'svelte/store';

const STORAGE_KEY = 'sneaker-matcher-auth';

const initialState = {
	token: null,
	user: null
};

export const auth = writable(initialState);

export function loadAuth() {
	if (!browser) return;
	const raw = localStorage.getItem(STORAGE_KEY);
	if (!raw) {
		auth.set(initialState);
		return;
	}

	try {
		auth.set(JSON.parse(raw));
	} catch {
		localStorage.removeItem(STORAGE_KEY);
		auth.set(initialState);
	}
}

export function saveAuth(token, user) {
	const next = { token, user };
	auth.set(next);
	if (browser) {
		localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
	}
}

export function clearAuth() {
	auth.set(initialState);
	if (browser) {
		localStorage.removeItem(STORAGE_KEY);
	}
}
