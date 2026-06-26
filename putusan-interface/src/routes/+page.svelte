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
	const modelFileCount = $derived(data.modelSummaries.filter((item) => item.count > 0).length);

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

	// Sections are split across two columns in the detail panel, mirroring the Figma layout.
	const sectionSplit = $derived(Math.ceil(data.columns.length / 2));
	const sectionsLeft = $derived(data.columns.slice(0, sectionSplit));
	const sectionsRight = $derived(data.columns.slice(sectionSplit));

	const preview = (value: string, limit = 180) => {
		const normalized = value.replace(/\s+/g, ' ').trim();
		return normalized.length > limit ? `${normalized.slice(0, limit)}...` : normalized || '-';
	};

	const resetFilters = () => {
		search = '';
		selectedCategory = 'all';
		selectedModel = 'all';
	};

	// --- Resizable table columns (drag the column border like Windows Explorer) ---
	const SOURCE_KEY = '__source__';
	const MIN_COL_WIDTH = 80;
	const DEFAULT_SOURCE_WIDTH = 180;
	const DEFAULT_COL_WIDTH = 170;
	// Only overridden widths are stored; everything else falls back to the defaults.
	let colWidths = $state<Record<string, number>>({});
	const widthOf = (key: string) =>
		colWidths[key] ?? (key === SOURCE_KEY ? DEFAULT_SOURCE_WIDTH : DEFAULT_COL_WIDTH);
	const tableWidth = $derived(
		[SOURCE_KEY, ...data.columns.map((column) => column.key)].reduce(
			(sum, key) => sum + widthOf(key),
			0
		)
	);

	let resizing: { key: string; startX: number; startWidth: number } | null = null;

	const onResizeMove = (event: PointerEvent) => {
		if (!resizing) return;
		const delta = event.clientX - resizing.startX;
		colWidths[resizing.key] = Math.max(MIN_COL_WIDTH, resizing.startWidth + delta);
	};

	const stopResize = () => {
		resizing = null;
		window.removeEventListener('pointermove', onResizeMove);
		window.removeEventListener('pointerup', stopResize);
	};

	const startResize = (event: PointerEvent, key: string) => {
		event.preventDefault();
		event.stopPropagation();
		resizing = { key, startX: event.clientX, startWidth: widthOf(key) };
		window.addEventListener('pointermove', onResizeMove);
		window.addEventListener('pointerup', stopResize);
	};

	// Keyboard fallback for resizing (accessibility): arrow keys nudge the column width.
	const nudgeWidth = (event: KeyboardEvent, key: string) => {
		if (event.key === 'ArrowLeft') {
			colWidths[key] = Math.max(MIN_COL_WIDTH, widthOf(key) - 16);
			event.preventDefault();
		} else if (event.key === 'ArrowRight') {
			colWidths[key] = widthOf(key) + 16;
			event.preventDefault();
		}
	};

	// --- Manual section verification ---
	// Checkboxes start EMPTY. A legal expert ticks each section they confirm exists
	// in the actual document. State is kept per row so each decision is independent,
	// and persisted server-side (POST /api/verified) so ticks are shared and survive
	// reloads. The server-loaded values seed the initial render; `overrides` holds
	// changes made this session and takes precedence.
	let overrides = $state<Record<string, boolean>>({});
	const verifyKey = (rowId: string, key: string) => `${rowId}::${key}`;
	const isVerified = (rowId: string, key: string) => {
		const k = verifyKey(rowId, key);
		return overrides[k] ?? data.verified[k] ?? false;
	};
	const toggleVerified = async (rowId: string, key: string) => {
		const k = verifyKey(rowId, key);
		const next = !isVerified(rowId, key);
		overrides[k] = next; // optimistic update
		try {
			const res = await fetch('/api/verified', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ key: k, value: next })
			});
			if (!res.ok) throw new Error(`save failed: ${res.status}`);
		} catch {
			overrides[k] = !next; // revert on failure so the UI stays honest
		}
	};
	const verifiedCount = $derived(
		selectedRow
			? data.columns.filter((column) => isVerified(selectedRow.id, column.key)).length
			: 0
	);
	// Review status reflects the expert's verification progress, NOT the JSON status.
	const allVerified = $derived(verifiedCount === data.columns.length);
	const reviewLabel = $derived(allVerified ? 'Selesai diverifikasi' : 'Belum selesai');
</script>

<svelte:head>
	<title>Putusan LLM Aggregator</title>
	<meta
		name="description"
		content="Interface untuk menampilkan hasil ekstraksi LLM Aggregator TPPO dan Anak."
	/>
</svelte:head>

<main class="min-h-screen bg-white text-[#26272f]">
	<div class="mx-auto w-full max-w-[1440px] px-[70px] py-12">
		<!-- Header -->
		<header class="flex flex-col gap-8 lg:flex-row lg:items-start lg:justify-between">
			<div class="flex max-w-[640px] flex-col gap-3">
				<h1 class="text-[40px] font-bold leading-tight text-[#26272f]">Putusan TPPO dan Anak</h1>
				<p class="text-[16px] leading-relaxed text-[#666]">
					Hasil JSON dari semua model ditampilkan sebagai baris putusan dengan struktur kolom yang
					mengikuti CSV referensi Human Trafficking Court Decision.
				</p>
			</div>

			<div class="flex flex-wrap gap-4">
				{#each [{ label: 'JSON rows', value: data.rows.length }, { label: 'Model files', value: modelFileCount }, { label: 'CSV rows', value: data.csvReference.rowCount }, { label: 'CSV columns', value: data.csvReference.columns.length }] as stat (stat.label)}
					<div
						class="flex h-[70px] min-w-[93px] flex-col items-center justify-center gap-1.5 rounded-[10px] bg-[#f3f5fe] px-2.5"
					>
						<span class="text-[12px] whitespace-nowrap text-[#666]">{stat.label}</span>
						<span class="text-[20px] font-bold text-[#26272f]">{stat.value}</span>
					</div>
				{/each}
			</div>
		</header>

		<!-- Search -->
		<div class="mt-12 flex items-center gap-3.5">
			<label for="search" class="text-[16px] font-bold text-[#666] capitalize">Search</label>
			<input
				id="search"
				type="search"
				bind:value={search}
				placeholder="Nomor putusan, pengadilan, file, amar putusan...."
				class="h-[34px] w-full max-w-[792px] rounded-[10px] border border-[#d6d6d6] bg-white px-2.5 text-[16px] text-[#26272f] placeholder:text-[#666] focus:border-[#26272f] focus:ring-0 focus:outline-none"
			/>
		</div>

		<!-- Content: rows table + detail panel -->
		<div class="mt-4 grid gap-6 lg:grid-cols-[minmax(0,1fr)_420px]">
			<!-- Rows table card -->
			<section
				class="flex min-w-0 flex-col rounded-[10px] border border-[#d6d6d6] bg-white"
			>
				<div class="flex items-start justify-between gap-4 px-4 pt-4 pb-3">
					<div class="flex flex-col gap-1">
						<h2 class="text-[16px] font-bold text-[#26272f]">Rows</h2>
						<p class="text-[12px] text-[#26272f]">{filteredRows.length} matching records</p>
					</div>
					<div class="flex items-center gap-2.5">
						<select
							bind:value={selectedCategory}
							class="h-[34px] rounded-[10px] border border-[#d6d6d6] bg-white px-2.5 text-[16px] text-[#26272f] focus:border-[#26272f] focus:ring-0 focus:outline-none"
						>
							{#each categories as category (category)}
								<option value={category}>{category === 'all' ? 'All Categories' : category}</option>
							{/each}
						</select>
						<select
							bind:value={selectedModel}
							class="h-[34px] rounded-[10px] border border-[#d6d6d6] bg-white px-2.5 text-[16px] text-[#26272f] focus:border-[#26272f] focus:ring-0 focus:outline-none"
						>
							{#each models as model (model)}
								<option value={model}>{model === 'all' ? 'All Models' : model}</option>
							{/each}
						</select>
						<button
							type="button"
							onclick={resetFilters}
							class="h-[34px] rounded-[10px] border border-[#d6d6d6] bg-[#ffcccc] px-2.5 text-[16px] text-[#26272f] transition hover:brightness-95"
						>
							Reset filters
						</button>
					</div>
				</div>

				<p class="px-4 pb-2 text-[12px] text-[#666]">
					Tip: tarik garis pemisah antar kolom untuk melebarkan atau mempersempit kolom.
				</p>
				<div class="max-h-[816px] overflow-auto">
					<table
						class="border-collapse text-left"
						style="table-layout:fixed; width:{tableWidth}px"
					>
						<colgroup>
							<col style="width:{widthOf(SOURCE_KEY)}px" />
							{#each data.columns as column (column.key)}
								<col style="width:{widthOf(column.key)}px" />
							{/each}
						</colgroup>
						<thead class="sticky top-0 z-10">
							<tr class="bg-[#f3f5fe]">
								<th
									class="relative border-r border-[#d6d6d6] px-4 py-5 text-[16px] font-bold tracking-wide text-[#666] uppercase"
								>
									Source
									<button
										type="button"
										aria-label="Resize Source column"
										onpointerdown={(event) => startResize(event, SOURCE_KEY)}
										onkeydown={(event) => nudgeWidth(event, SOURCE_KEY)}
										class="absolute top-0 right-0 z-20 h-full w-2 cursor-col-resize hover:bg-[#9ca3af]/60 focus:bg-[#9ca3af]/60 focus:outline-none"
									></button>
								</th>
								{#each data.columns as column (column.key)}
									<th
										class="relative border-r border-[#d6d6d6] px-3 py-5 text-[16px] font-bold tracking-wide text-[#666] uppercase last:border-r-0"
									>
										{column.label}
										<button
											type="button"
											aria-label={`Resize ${column.label} column`}
											onpointerdown={(event) => startResize(event, column.key)}
											onkeydown={(event) => nudgeWidth(event, column.key)}
											class="absolute top-0 right-0 z-20 h-full w-2 cursor-col-resize hover:bg-[#9ca3af]/60 focus:bg-[#9ca3af]/60 focus:outline-none"
										></button>
									</th>
								{/each}
							</tr>
						</thead>
						<tbody>
							{#each filteredRows as row (row.id)}
								<tr
									onclick={() => (selectedRowId = row.id)}
									class="cursor-pointer border-b border-[#eef0f3] align-top transition hover:bg-[#f3f5fe]"
									class:bg-[#ddefff]={selectedRow?.id === row.id}
								>
									<td class="border-r border-[#eef0f3] px-4 py-5 align-top">
										<div class="flex flex-col gap-2.5">
											<div class="flex flex-col gap-1">
												<p class="text-[16px] font-bold text-[#26272f] capitalize">
													{row.category} / {row.model}
												</p>
												<p class="text-[16px] break-all text-[#666]">{row.fileName}</p>
											</div>
											<span
												class="inline-flex w-fit items-center rounded-full border border-[#666] px-2 py-0.5 text-[12px] text-[#666]"
												class:bg-[#d5ffd2]={row.status.toLowerCase() === 'completed'}
												class:bg-[#d6d6d6]={row.status.toLowerCase() !== 'completed'}
											>
												{row.status}
											</span>
										</div>
									</td>
									{#each data.columns as column (column.key)}
										<td
											class="border-r border-[#eef0f3] px-3 py-5 text-[16px] break-words text-[#26272f] last:border-r-0"
										>
											{preview(row.fields[column.key] ?? '', 120)}
										</td>
									{/each}
								</tr>
							{:else}
								<tr>
									<td
										class="px-4 py-12 text-center text-[16px] text-[#666]"
										colspan={data.columns.length + 1}
									>
										No rows match the current filters.
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</section>

			<!-- Detail panel -->
			<aside
				class="flex min-w-0 flex-col rounded-[10px] border border-[#d6d6d6] bg-white lg:sticky lg:top-6 lg:self-start"
			>
				{#if selectedRow}
					<!-- Selected header -->
					<div class="flex flex-col gap-1.5 px-5 pt-5 pb-4">
						<p class="text-[16px] font-bold text-[#acd2f3]">Selected Items</p>
						<p class="text-[20px] font-bold text-[#26272f] capitalize">
							{selectedRow.fields.nomor_putusan || selectedRow.fileName}
						</p>
						<p class="text-[16px] text-[#666] capitalize">
							{selectedRow.category} / {selectedRow.model} / {selectedRow.sourceFile}
						</p>
					</div>
					<hr class="border-[#d6d6d6]" />

					<!-- Meta grid -->
					<div class="grid grid-cols-2 gap-x-8 gap-y-4 px-5 py-5">
						<div class="flex flex-col gap-2.5">
							<p class="text-[12px] font-bold text-[#666]">Pengadilan</p>
							<p class="text-[13px] text-[#26272f]">
								{preview(selectedRow.fields.nama_pengadilan_negeri, 90)}
							</p>
						</div>
						<div class="flex flex-col gap-2.5">
							<p class="text-[12px] font-bold text-[#666]">Tahun</p>
							<p class="text-[13px] text-[#26272f]">{preview(selectedRow.fields.tahun, 40)}</p>
						</div>
						<div class="flex flex-col gap-2.5">
							<p class="text-[12px] font-bold text-[#666]">Empty sections</p>
							<p class="text-[13px] text-[#26272f] capitalize">
								{selectedRow.emptySections.length
									? selectedRow.emptySections.join(', ')
									: 'None'}
							</p>
						</div>
						<div class="flex flex-col gap-2.5">
							<p class="text-[12px] font-bold text-[#666]">Review status</p>
							<div class="flex items-center justify-between gap-2">
								<span
									class="text-[13px] font-bold"
									class:text-[#15803d]={allVerified}
									class:text-[#b45309]={!allVerified}
								>
									{reviewLabel}
								</span>
								<span
									class="flex size-[14px] items-center justify-center rounded-[3px] border text-[11px] leading-none"
									class:border-[#15803d]={allVerified}
									class:bg-[#15803d]={allVerified}
									class:text-white={allVerified}
									class:border-[#d6d6d6]={!allVerified}
									class:bg-white={!allVerified}
								>
									{allVerified ? '✓' : ''}
								</span>
							</div>
						</div>
					</div>
					<hr class="border-[#d6d6d6]" />

					<!-- Sections checklist (manual verification by a legal expert) -->
					<div class="flex flex-col gap-3 px-5 py-5">
						<div class="flex items-center justify-between gap-2">
							<p class="text-[12px] font-bold text-[#666]">Sections</p>
							<span class="text-[12px] font-bold text-[#26272f]">
								{verifiedCount}/{data.columns.length} dicentang
							</span>
						</div>
						<p class="text-[12px] leading-snug text-[#666]">
							Centang bagian yang benar-benar ada di dokumen putusan.
						</p>
						<div class="grid grid-cols-2 gap-x-6">
							{#each [sectionsLeft, sectionsRight] as columnGroup, groupIndex (groupIndex)}
								<div class="flex flex-col gap-0.5">
									{#each columnGroup as column (column.key)}
										<label
											class="flex cursor-pointer items-center justify-between gap-2 rounded-md px-1.5 py-1.5 hover:bg-[#f3f5fe]"
										>
											<span class="text-[13px] font-bold text-[#26272f] capitalize select-none">
												{column.label}
											</span>
											<input
												type="checkbox"
												checked={isVerified(selectedRow.id, column.key)}
												onchange={() => selectedRow && toggleVerified(selectedRow.id, column.key)}
												class="size-[18px] shrink-0 cursor-pointer rounded border-[#9ca3af] text-[#2563eb] focus:ring-[#2563eb]"
											/>
										</label>
									{/each}
								</div>
							{/each}
						</div>
					</div>
					<hr class="border-[#d6d6d6]" />

					<!-- JSON view -->
					<div class="flex flex-col gap-2.5 px-5 py-5">
						<p class="text-[12px] font-bold text-[#666]">Json view</p>
						<div class="max-h-[354px] overflow-auto rounded-[9px] bg-[#26272f] p-4">
							<pre
								class="text-[12px] leading-5 whitespace-pre-wrap break-words text-slate-100">{JSON.stringify(
									selectedRow.json,
									null,
									2
								)}</pre>
						</div>
					</div>
				{:else}
					<p class="px-5 py-12 text-[16px] text-[#666]">Select a row to inspect its details.</p>
				{/if}
			</aside>
		</div>
	</div>
</main>
