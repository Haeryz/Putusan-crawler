import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';

import { readVerified } from '$lib/server/verifiedStore';
import type { PageServerLoad } from './$types';

type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };
type JsonObject = { [key: string]: JsonValue };

type DecisionRow = {
	id: string;
	category: string;
	model: string;
	fileName: string;
	sourceFile: string;
	status: string;
	emptySections: string[];
	json: JsonObject;
	fields: Record<string, string>;
};

type ModelSummary = {
	category: string;
	model: string;
	count: number;
	bytes: number;
};

const categoryNames = ['TPPO', 'Anak'];
const modelNames = ['Deepseek', 'Gemini', 'GPT', 'Qwen'];

const rowColumns = [
	{ key: 'judul', label: 'Judul' },
	{ key: 'nomor_putusan', label: 'Nomor putusan' },
	{ key: 'irah_irah', label: 'Irah-irah' },
	{ key: 'nama_pengadilan_negeri', label: 'Nama pengadilan negeri' },
	{ key: 'keterangan_perkara', label: 'Keterangan perkara' },
	{ key: 'identitas_terdakwa', label: 'Identitas terdakwa' },
	{ key: 'penangkapan', label: 'Penangkapan' },
	{ key: 'penahanan', label: 'Penahanan' },
	{ key: 'tuntutan', label: 'Tuntutan' },
	{ key: 'dakwaan', label: 'Dakwaan' },
	{ key: 'saksi', label: 'Saksi' },
	{ key: 'ahli', label: 'Ahli' },
	{ key: 'terdakwa', label: 'Terdakwa' },
	{ key: 'surat', label: 'Surat' },
	{ key: 'petunjuk_barang_bukti', label: 'Petunjuk/BB' },
	{ key: 'fakta_hukum', label: 'Fakta hukum' },
	{ key: 'pertimbangan_hukum', label: 'Pertimbangan hukum' },
	{ key: 'amar_putusan', label: 'Amar putusan' },
	{ key: 'hari', label: 'Hari' },
	{ key: 'tanggal', label: 'Tanggal' },
	{ key: 'tahun', label: 'Tahun' },
	{ key: 'siapa_yang_memutus', label: 'Siapa yang memutus' },
	{ key: 'panitera_pengganti', label: 'Panitera pengganti' },
	{ key: 'tanda_tangan_majelis', label: 'Tanda tangan majelis' }
];

const identityKeys = [
	'nama_lengkap',
	'tempat_lahir',
	'umur_tanggal_lahir',
	'jenis_kelamin',
	'kebangsaan',
	'tempat_tinggal',
	'agama',
	'pekerjaan'
];

const sectionAliases: Record<string, string[]> = {
	petunjuk_barang_bukti: ['petunjuk_barang_bukti', 'petunjuk_bb', 'petunjuk/BB', 'petunjuk/ bb'],
	siapa_yang_memutus: ['siapa_yang_memutus', 'majelis_hakim', 'hakim'],
	panitera_pengganti: ['panitera_pengganti', 'panitera'],
	identitas_terdakwa: ['identitas_terdakwa']
};

const toDisplayText = (value: JsonValue | undefined): string => {
	if (value === undefined || value === null) return '';
	if (Array.isArray(value))
		return value
			.map((item) => toDisplayText(item))
			.filter(Boolean)
			.join('\n\n');
	if (typeof value === 'object') return JSON.stringify(value, null, 2);
	return String(value);
};

const getSectionText = (sections: JsonObject, key: string): string => {
	const aliases = sectionAliases[key] ?? [key];

	for (const alias of aliases) {
		const text = toDisplayText(sections[alias]);
		if (text) return text;
	}

	if (key === 'identitas_terdakwa') {
		return identityKeys
			.map((identityKey) => {
				const text = getSectionText(sections, identityKey);
				return text ? `${identityKey.replaceAll('_', ' ')}: ${text}` : '';
			})
			.filter(Boolean)
			.join('\n');
	}

	return '';
};

const normalizeRow = (
	json: JsonObject,
	fileName: string,
	category: string,
	model: string
): DecisionRow => {
	const sections =
		typeof json.sections === 'object' && json.sections !== null
			? (json.sections as JsonObject)
			: {};
	const fields = Object.fromEntries(
		rowColumns.map((column) => [column.key, getSectionText(sections, column.key)])
	);
	const sourceFile = toDisplayText(json.source_file) || fileName.replace(/\.json$/i, '.txt');
	const emptySections = Array.isArray(json.empty_sections)
		? json.empty_sections.map((section) => String(section))
		: [];

	return {
		id: `${category}/${model}/${fileName}`,
		category,
		model,
		fileName,
		sourceFile,
		status: toDisplayText(json.status) || 'unknown',
		emptySections,
		json,
		fields
	};
};

const collectJsonFiles = async (directory: string): Promise<string[]> => {
	const entries = await readdir(directory, { withFileTypes: true });
	const files = await Promise.all(
		entries.map(async (entry) => {
			const entryPath = path.join(directory, entry.name);
			if (entry.isDirectory()) return collectJsonFiles(entryPath);
			return entry.isFile() && entry.name.toLowerCase().endsWith('.json') ? [entryPath] : [];
		})
	);

	return files.flat();
};

const readModelRows = async (
	root: string,
	category: string,
	model: string
): Promise<{ rows: DecisionRow[]; summary: ModelSummary }> => {
	const outputDirectory = path.join(root, category, model, 'output');
	const summary: ModelSummary = { category, model, count: 0, bytes: 0 };

	try {
		await stat(outputDirectory);
	} catch {
		return { rows: [], summary };
	}

	const jsonFiles = await collectJsonFiles(outputDirectory);
	const rows: DecisionRow[] = [];

	for (const filePath of jsonFiles) {
		const content = await readFile(filePath, 'utf8');
		summary.bytes += Buffer.byteLength(content, 'utf8');
		const json = JSON.parse(content.replace(/^\uFEFF/, '')) as JsonObject;
		rows.push(normalizeRow(json, path.basename(filePath), category, model));
	}

	rows.sort((a, b) => a.fileName.localeCompare(b.fileName, 'id'));
	summary.count = rows.length;
	return { rows, summary };
};

const readCsvReference = async (csvPath: string) => {
	try {
		const csv = await readFile(csvPath, 'utf8');
		const [header = ''] = csv.split(/\r?\n/, 1);
		const rowCount = csv.split(/\r?\n/).filter(Boolean).length - 1;

		return {
			path: csvPath,
			rowCount: Math.max(0, rowCount),
			columns: header
				.split(';')
				.map((column) => column.trim())
				.filter(Boolean)
		};
	} catch {
		return {
			path: csvPath,
			rowCount: 0,
			columns: []
		};
	}
};

export const load: PageServerLoad = async () => {
	const workspaceRoot = path.resolve(process.cwd(), '..');
	const aggregatorRoot = path.join(workspaceRoot, 'LLM-aggregator');
	const csvPath = path.join(workspaceRoot, 'data', 'Human_Trafficking_Court_Decission_utf8.csv');

	const results = await Promise.all(
		categoryNames.flatMap((category) =>
			modelNames.map((model) => readModelRows(aggregatorRoot, category, model))
		)
	);

	return {
		csvReference: await readCsvReference(csvPath),
		columns: rowColumns,
		modelSummaries: results.map((result) => result.summary),
		rows: results.flatMap((result) => result.rows),
		verified: await readVerified()
	};
};
