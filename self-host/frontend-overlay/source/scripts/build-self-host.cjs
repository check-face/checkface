// Build the original Elmish UI only; same-origin legacy API, no trial or analytics.
const { spawnSync } = require('node:child_process');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const env = { ...process.env, FACEMORPH_SELF_HOST: '1', FACEMORPH_TRIAL: '0', FACEMORPH_REVIEW: '0', FACEMORPH_TRIAL_URL: '' };
for (const [command, args] of [
  ['dotnet', ['tool', 'restore']],
  ['dotnet', ['fable', './src']],
  [process.execPath, [require.resolve('webpack-cli/bin/cli.js'), '--config', 'webpack.config.js']],
]) {
  const result = spawnSync(command, args, { cwd: root, env, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status || 1);
}
