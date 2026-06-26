import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

/**
 * Server-side persistence for manual section verifications.
 *
 * Stored as a flat map of `${rowId}::${sectionKey}` -> boolean in a JSON file
 * inside the project's `.data/` directory. Writes are serialized so concurrent
 * toggles can't clobber each other (read-modify-write under a single chain).
 */

export type VerifiedMap = Record<string, boolean>;

const dataDir = path.resolve(process.cwd(), '.data');
const filePath = path.join(dataDir, 'verified-sections.json');

let chain: Promise<unknown> = Promise.resolve();
const runExclusive = <T>(task: () => Promise<T>): Promise<T> => {
	const result = chain.then(task, task);
	chain = result.then(
		() => undefined,
		() => undefined
	);
	return result;
};

export const readVerified = async (): Promise<VerifiedMap> => {
	try {
		const raw = await readFile(filePath, 'utf8');
		const parsed = JSON.parse(raw) as unknown;
		return parsed && typeof parsed === 'object' ? (parsed as VerifiedMap) : {};
	} catch {
		return {};
	}
};

const persist = async (map: VerifiedMap): Promise<void> => {
	await mkdir(dataDir, { recursive: true });
	await writeFile(filePath, JSON.stringify(map, null, 2), 'utf8');
};

/** Set a single verification flag and return the full updated map. */
export const setVerified = (key: string, value: boolean): Promise<VerifiedMap> =>
	runExclusive(async () => {
		const current = await readVerified();
		if (value) current[key] = true;
		else delete current[key];
		await persist(current);
		return current;
	});
