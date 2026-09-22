#!/usr/bin/env node

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, copyFileSync, chmodSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');

function hostTriple() {
  const out = execFileSync('rustc', ['-vV'], { encoding: 'utf8' });
  const match = out.match(/^host:\s*(\S+)$/m);
  if (!match) {
    throw new Error('Could not determine host triple from `rustc -vV`');
  }
  return match[1];
}

const triple = hostTriple();
const isWindows = process.platform === 'win32';
const exeSuffix = isWindows ? '.exe' : '';

import { statSync } from 'node:fs';

export function selectBinary(candidates, existsSyncFn, statSyncFn) {
  const existingCandidates = candidates.filter((p) => existsSyncFn(p));
  if (existingCandidates.length === 0) {
    throw new Error(`Sidecar binary not found in ${candidates.join(' or ')}.`);
  }
  let srcBinary = existingCandidates[0];
  if (existingCandidates.length > 1) {
    const stats = existingCandidates.map((p) => ({ path: p, mtime: statSyncFn(p).mtimeMs }));
    stats.sort((a, b) => b.mtime - a.mtime);
    srcBinary = stats[0].path;
  }
  return srcBinary;
}

const isDirectRun = process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url));

if (isDirectRun) {
  const triple = hostTriple();
  const isWindows = process.platform === 'win32';
  const exeSuffix = isWindows ? '.exe' : '';

  const destDir = join(repoRoot, 'src-tauri', 'binaries');
  mkdirSync(destDir, { recursive: true });

  const llmCandidates = [
    join(repoRoot, 'target', 'release', `fotonvoice-llm-sidecar${exeSuffix}`),
    join(repoRoot, 'target', 'debug', `fotonvoice-llm-sidecar${exeSuffix}`),
  ];
  const existingLlm = llmCandidates.filter((p) => existsSync(p));
  if (existingLlm.length > 0) {
    let srcLlm = existingLlm[0];
    if (existingLlm.length > 1) {
      const stats = existingLlm.map((p) => ({ path: p, mtime: statSync(p).mtimeMs }));
      stats.sort((a, b) => b.mtime - a.mtime);
      srcLlm = stats[0].path;
    }
    const destLlm = join(destDir, `fotonvoice-llm-sidecar-${triple}${exeSuffix}`);
    copyFileSync(srcLlm, destLlm);
    if (!isWindows) {
      chmodSync(destLlm, 0o755);
    }
    console.log(`[prepare-sidecar] staged ${srcLlm} -> ${destLlm}`);
  } else {
    console.warn(`[prepare-sidecar] Note: fotonvoice-llm-sidecar binary not found yet in target/release or target/debug`);
  }
}
