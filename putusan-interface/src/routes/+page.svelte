<script lang="ts">
	import type { PageData } from './$types';

	type Row = PageData['rows'][number];

	let { data }: { data: PageData } = $props();

	let search = $state('');
	let selectedCategory = $state('all');
	let selectedModel = $state('all');
	let selectedRowId = $state('');

	const categories = $derived([
		'all',
		...new Set(data.modelSummaries.map((item) => item.category))
	]);
	const models = $derived(['all', ...new Set(data.modelSummaries.map((item) => item.model))]);
	const visibleColumns = $derived(data.columns.slice(0, 10));
	const totalBytes = $derived(data.modelSummaries.reduce((sum, item) => sum + item.bytes, 0));

	const filteredRows = $derived(
		data.rows.filter((row) => {
			const query = search.trim().toLowerCase();
			const matchesCategory = selectedCategory === 'all' || row.category === selectedCategory;
			const matchesModel = selectedModel === 'all' || row.model === selectedModel;
			const haystack = [
				row.category,
				row.model,
				row.fileName,
				row.sourceFile,
				row.status,
				row.fields.nomor_putusan,
				row.fields.nama_pengadilan_negeri,
				row.fields.amar_putusan
			]
				.join(' ')
				.toLowerCase();

			return matchesCategory && matchesModel && (!query || haystack.includes(query));
		})
	);

	const selectedRow = $derived<Row | undefined>(
		filteredRows.find((row) => row.id === selectedRowId) ?? filteredRows[0] ?? data.rows[0]
	);

	const formatBytes = (bytes: number) => {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
	};

	const preview = (value: string, limit = 180) => {
		const normalized = value.replace(/\s+/g, ' ').trim();
		return normalized.length > limit ? `${normalized.slice(0, limit)}...` : normalized || '-';
	};
</script>

<svelte:head>
	<title>Putusan LLM Aggregator</title>
	<meta
		name="description"
		content="Interface untuk menampilkan hasil ekstraksi LLM Aggregator TPPO dan Anak."
	/>
</svelte:head>

<main class="min-h-screen bg-slate-50 text-slate-950">
	<section class="border-b border-slate-200 bg-white">
		<div class="mx-auto flex max-w-[1800px] flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
			<div class="flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
				<div>
					<p class="text-sm font-semibold uppercase tracking-wide text-cyan-700">LLM Aggregator</p>
					<h1 class="mt-2 text-3xl font-semibold tracking-normal text-slate-950">
						Putusan TPPO dan Anak
					</h1>
					<p class="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
						Hasil JSON dari semua model ditampilkan sebagai baris putusan dengan struktur kolom yang
						mengikuti CSV referensi Human Trafficking Court Decision.
					</p>
				</div>

				<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
					<div class="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
						<p class="text-xs font-medium text-slate-500">JSON rows</p>
						<p class="mt-1 text-2xl font-semibold">{data.rows.length}</p>
					</div>
					<div class="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
						<p class="text-xs font-medium text-slate-500">Model files</p>
						<p class="mt-1 text-2xl font-semibold">{formatBytes(totalBytes)}</p>
					</div>
					<div class="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
						<p class="text-xs font-medium text-slate-500">CSV rows</p>
						<p class="mt-1 text-2xl font-semibold">{data.csvReference.rowCount}</p>
					</div>
					<div class="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
						<p class="text-xs font-medium text-slate-500">CSV columns</p>
						<p class="mt-1 text-2xl font-semibold">{data.csvReference.columns.length}</p>
					</div>
				</div>
			</div>

			<div class="grid gap-3 lg:grid-cols-[minmax(260px,1fr)_180px_180px]">
				<label class="block">
					<span class="text-xs font-medium text-slate-600">Search</span>
					<input
						class="mt-1 w-full rounded-lg border-slate-300 bg-white text-sm shadow-sm focus:border-cyan-600 focus:ring-cyan-600"
						type="search"
						bind:value={search}
						placeholder="Nomor putusan, pengadilan, file, amar putusan..."
					/>
				</label>

				<label class="block">
					<span class="text-xs font-medium text-slate-600">Category</span>
					<select
						class="mt-1 w-full rounded-lg border-slate-300 bg-white text-sm shadow-sm focus:border-cyan-600 focus:ring-cyan-600"
						bind:value={selectedCategory}
					>
						{#each categories as category (category)}
							<option value={category}>{category === 'all' ? 'All categories' : category}</option>
						{/each}
					</select>
				</label>

				<label class="block">
					<span class="text-xs font-medium text-slate-600">Model</span>
					<select
						class="mt-1 w-full rounded-lg border-slate-300 bg-white text-sm shadow-sm focus:border-cyan-600 focus:ring-cyan-600"
						bind:value={selectedModel}
					>
						{#each models as model (model)}
							<option value={model}>{model === 'all' ? 'All models' : model}</option>
						{/each}
					</select>
				</label>
			</div>
		</div>
	</section>

	<section
		class="mx-auto grid max-w-[1800px] gap-5 px-4 py-5 sm:px-6 lg:grid-cols-[minmax(0,1fr)_440px] lg:px-8"
	>
		<div class="min-w-0 space-y-5">
			<div class="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
				{#each data.modelSummaries as summary (`${summary.category}-${summary.model}`)}
					<button
						class="rounded-lg border border-slate-200 bg-white px-4 py-3 text-left shadow-sm transition hover:border-cyan-500 hover:bg-cyan-50"
						type="button"
						onclick={() => {
							selectedCategory = summary.category;
							selectedModel = summary.model;
						}}
					>
						<div class="flex items-center justify-between gap-3">
							<p class="font-semibold">{summary.category} / {summary.model}</p>
							<span class="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700"
								>{summary.count}</span
							>
						</div>
						<p class="mt-2 text-xs text-slate-500">{formatBytes(summary.bytes)} JSON output</p>
					</button>
				{/each}
			</div>

			<div class="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
				<div
					class="flex flex-col gap-2 border-b border-slate-200 px-4 py-3 sm:flex-row sm:items-center sm:justify-between"
				>
					<div>
						<h2 class="text-base font-semibold">Rows</h2>
						<p class="text-sm text-slate-500">{filteredRows.length} matching records</p>
					</div>
					<button
						class="w-fit rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
						type="button"
						onclick={() => {
							search = '';
							selectedCategory = 'all';
							selectedModel = 'all';
						}}
					>
						Reset filters
					</button>
				</div>

				<div class="max-h-[68vh] overflow-auto">
					<table class="min-w-[1440px] divide-y divide-slate-200 text-left text-sm">
						<thead class="sticky top-0 z-10 bg-slate-100 text-xs uppercase text-slate-600">
							<tr>
								<th class="w-44 px-3 py-3 font-semibold">Source</th>
								{#each visibleColumns as column (column.key)}
									<th class="px-3 py-3 font-semibold">{column.label}</th>
								{/each}
							</tr>
						</thead>
						<tbody class="divide-y divide-slate-100">
							{#each filteredRows as row (row.id)}
								<tr
									class:bg-cyan-50={selectedRow?.id === row.id}
									class="cursor-pointer align-top hover:bg-slate-50"
									onclick={() => (selectedRowId = row.id)}
								>
									<td class="px-3 py-3">
										<p class="font-semibold text-slate-950">{row.category} / {row.model}</p>
										<p class="mt-1 break-all text-xs text-slate-500">{row.fileName}</p>
										<p
											class="mt-2 inline-flex rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600"
										>
											{row.status}
										</p>
									</td>
									{#each visibleColumns as column (column.key)}
										<td class="max-w-[280px] px-3 py-3 text-slate-700">
											{preview(row.fields[column.key] ?? '')}
										</td>
									{/each}
								</tr>
							{:else}
								<tr>
									<td
										class="px-4 py-10 text-center text-slate-500"
										colspan={visibleColumns.length + 1}
									>
										No rows match the current filters.
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		</div>

		<aside class="min-w-0 space-y-5 lg:sticky lg:top-5 lg:self-start">
			<div class="rounded-lg border border-slate-200 bg-white shadow-sm">
				<div class="border-b border-slate-200 px-4 py-3">
					<p class="text-xs font-semibold uppercase tracking-wide text-cyan-700">Selected JSON</p>
					<h2 class="mt-1 break-words text-lg font-semibold">
						{selectedRow?.fields.nomor_putusan || selectedRow?.fileName || 'No record selected'}
					</h2>
					{#if selectedRow}
						<p class="mt-1 text-sm text-slate-500">
							{selectedRow.category} / {selectedRow.model} / {selectedRow.sourceFile}
						</p>
					{/if}
				</div>

				{#if selectedRow}
					<div class="grid grid-cols-2 gap-3 border-b border-slate-200 px-4 py-4 text-sm">
						<div>
							<p class="text-xs font-medium text-slate-500">Pengadilan</p>
							<p class="mt-1 font-medium">
								{preview(selectedRow.fields.nama_pengadilan_negeri, 90)}
							</p>
						</div>
						<div>
							<p class="text-xs font-medium text-slate-500">Tanggal</p>
							<p class="mt-1 font-medium">
								{preview(
									[
										selectedRow.fields.hari,
										selectedRow.fields.tanggal,
										selectedRow.fields.tahun
									].join(' '),
									90
								)}
							</p>
						</div>
						<div class="col-span-2">
							<p class="text-xs font-medium text-slate-500">Empty sections</p>
							<p class="mt-1 font-medium">
								{selectedRow.emptySections.length ? selectedRow.emptySections.join(', ') : 'None'}
							</p>
						</div>
					</div>

					<div class="max-h-[58vh] overflow-auto bg-slate-950 p-4">
						<pre
							class="whitespace-pre-wrap break-words text-xs leading-5 text-slate-100">{JSON.stringify(
								selectedRow.json,
								null,
								2
							)}</pre>
					</div>
				{:else}
					<p class="px-4 py-10 text-sm text-slate-500">
						Select a row to inspect its original JSON.
					</p>
				{/if}
			</div>

			<div class="rounded-lg border border-slate-200 bg-white px-4 py-4 shadow-sm">
				<h2 class="text-base font-semibold">CSV reference</h2>
				<p class="mt-2 break-all text-sm text-slate-500">{data.csvReference.path}</p>
				<div class="mt-3 flex flex-wrap gap-2">
					{#each data.csvReference.columns.slice(0, 12) as column (column)}
						<span class="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700">
							{column}
						</span>
					{/each}
				</div>
			</div>
		</aside>
	</section>
</main>
