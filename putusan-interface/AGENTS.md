# Repository Guidelines

## Project Structure & Module Organization

This repository is a SvelteKit frontend application. Source code lives in `src/`: reusable code belongs in `src/lib/`, route pages and layouts live in `src/routes/`, and global app shell files are `src/app.html` and `src/app.d.ts`. Static public assets should go in `static/`; bundled library assets currently live under `src/lib/assets/`. Generated build output such as `.svelte-kit/` and dependency folders such as `node_modules/` should not be committed.

## Build, Test, and Development Commands

- `bun install`: install dependencies from `package.json` and `bun.lock`.
- `bun run dev`: start the Vite development server.
- `bun run build`: create a production build with Vite/SvelteKit.
- `bun run preview`: serve the production build locally for review.
- `bun run check`: run `svelte-check` with the project TypeScript config.
- `bun run lint`: verify Prettier formatting and run ESLint.
- `bun run format`: format the workspace with Prettier.
- `bun run test:unit`: run Vitest unit and component tests.
- `bun run test:e2e`: install Playwright browsers if needed and run end-to-end tests.
- `bun run test`: run unit tests once, then Playwright end-to-end tests.

## Coding Style & Naming Conventions

Use TypeScript and Svelte conventions already present in the project. Keep components focused and place shared helpers or components in `src/lib/`. Use 2-space indentation as enforced by Prettier, and let `prettier-plugin-svelte` and `prettier-plugin-tailwindcss` order markup and Tailwind classes. Name Svelte components in `PascalCase.svelte`; name tests after the unit under test, for example `Welcome.svelte.spec.ts` or `greet.spec.ts`.

## Testing Guidelines

Tests use Vitest for unit/component coverage and Playwright for browser-level behavior. Add deterministic tests next to the code or in the relevant feature folder. Prefer testing public behavior over implementation details. Run `bun run check`, `bun run lint`, and the relevant test command before submitting changes; run `bun run test` for full local validation.

## Commit & Pull Request Guidelines

Recent history mixes concise descriptive subjects and conventional `feat:` prefixes. Use short, specific commit messages such as `feat: add search filters` or `Refactor validation reporting`. Avoid placeholder commit text. Pull requests should include a concise summary, the commands run, linked issues or context, and screenshots only when UI behavior or visual layout changes.

## Security & Configuration Tips

Do not commit local environment files, generated build artifacts, Playwright reports, or dependency directories. Keep secrets out of route code and tests. Prefer documented configuration through SvelteKit environment handling when adding runtime settings.
