import { json } from '@sveltejs/kit';

import { readVerified, setVerified } from '$lib/server/verifiedStore';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => {
	return json({ verified: await readVerified() });
};

export const POST: RequestHandler = async ({ request }) => {
	let body: { key?: unknown; value?: unknown };
	try {
		body = await request.json();
	} catch {
		return json({ error: 'Invalid JSON body' }, { status: 400 });
	}

	const { key, value } = body;
	if (typeof key !== 'string' || key.length === 0) {
		return json({ error: 'A non-empty string "key" is required' }, { status: 400 });
	}

	const verified = await setVerified(key, Boolean(value));
	return json({ ok: true, verified });
};
