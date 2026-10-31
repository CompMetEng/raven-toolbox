function generate_oracles()
% generate_oracles  Record MATLAB RAVEN's answers for the parity checks.
%
% Runs a set of RAVEN functions on the fixtures in tests/data/parity/ and
% writes what they return as JSON into tests/data/parity/oracles/. The Python
% side (tests/parity/test_oracles.py) then asserts that raven-toolbox returns
% the same, without needing MATLAB at test time.
%
% Only tier-1 behaviour belongs here: deterministic answers that both
% implementations should agree on exactly. Anything solved by MILP has
% alternate optima and is compared as an overlap band instead -- see
% tests/parity/README.md.
%
% Usage, with RAVEN on the MATLAB path:
%
%     cd scripts/parity
%     generate_oracles
%
% Commit the resulting JSON together with the RAVEN commit it came from, which
% is recorded in oracles/PROVENANCE.md by this function.

here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
fixtureDir = fullfile(repoRoot, 'tests', 'data', 'parity');
oracleDir = fullfile(fixtureDir, 'oracles');
if ~exist(oracleDir, 'dir')
    mkdir(oracleDir);
end

model = readYAMLmodel(fullfile(fixtureDir, 'tiny.yml'));

writeOracle(oracleDir, 'model_structure', modelStructure(model));
writeOracle(oracleDir, 'elemental_balance', elementalBalance(model));
writeOracle(oracleDir, 'gpr_dnf', gprDnf(model));
writeOracle(oracleDir, 'reporter_fold_changes', reporterFoldChanges(model));
writeOracle(oracleDir, 'empty_condition', emptyCondition(model));

writeProvenance(oracleDir);
fprintf('Oracles written to %s\n', oracleDir);
end


function out = modelStructure(model)
% What the reader saw: ids, bounds, stoichiometry, gene rules.
out = struct();
out.rxns = reshape(model.rxns, 1, []);
out.mets = reshape(model.mets, 1, []);
out.genes = reshape(model.genes, 1, []);
out.lb = reshape(model.lb, 1, []);
out.ub = reshape(model.ub, 1, []);
out.grRules = reshape(model.grRules, 1, []);

% Stoichiometry as a list of {rxn, met, coefficient}, so the comparison does
% not depend on either side's matrix layout.
[metIdx, rxnIdx, coeff] = find(model.S);
entries = cell(1, numel(coeff));
for k = 1:numel(coeff)
    entries{k} = struct( ...
        'rxn', model.rxns{rxnIdx(k)}, ...
        'met', model.mets{metIdx(k)}, ...
        'coefficient', coeff(k));
end
out.stoichiometry = entries;
end


function out = elementalBalance(model)
% getElementalBalance grades each reaction: 1 balanced, 0 unbalanced,
% -1 undecidable (a participating metabolite has no formula).
balance = getElementalBalance(model);
out = struct();
out.rxns = reshape(model.rxns, 1, []);
out.balanceStatus = reshape(balance.balanceStatus, 1, []);
end


function out = gprDnf(model)
% Gene rules normalised to disjunctive normal form.
rules = cell(1, numel(model.grRules));
for i = 1:numel(model.grRules)
    if isempty(model.grRules{i})
        rules{i} = '';
    else
        rules{i} = grRuleToDNF(model.grRules{i});
    end
end
out = struct();
out.rxns = reshape(model.rxns, 1, []);
out.grRules = reshape(model.grRules, 1, []);
out.dnf = rules;
end


function out = reporterFoldChanges(model)
% Reporter metabolites for the full gene set and for the up- and
% down-regulated subsets, with one fold change per gene.
pvalues = [0.01 0.2 0.7];
foldChanges = [2 -1 3];
if numel(model.genes) ~= numel(pvalues)
    error('generate_oracles:fixture', 'tiny.yml must have %d genes.', numel(pvalues));
end
rep = reporterMetabolites(model, model.genes, pvalues, 'geneFoldChanges', foldChanges);
tests = cell(1, numel(rep));
for k = 1:numel(rep)
    tests{k} = struct('test', rep(k).test, 'mets', {reshape(rep(k).mets, 1, [])}, ...
        'metZScores', reshape(rep(k).metZScores, 1, []));
end
out = struct();
out.genes = reshape(model.genes, 1, []);
out.pvalues = pvalues;
out.foldChanges = foldChanges;
out.tests = tests;
end


function out = emptyCondition(model)
% An empty condition file: how many fields parseYAML returns and whether
% applyCondition changes the bounds.
file = [tempname '.yml'];
fid = fopen(file, 'w');
fclose(fid);
cleanup = onCleanup(@() delete(file));
cond = parseYAML(file);
changed = applyCondition(model, file);
out = struct();
out.nFields = numel(fieldnames(cond));
out.modelChanged = ~isequal(model.lb, changed.lb) || ~isequal(model.ub, changed.ub);
end


function writeOracle(oracleDir, name, data)
target = fullfile(oracleDir, [name '.json']);
fid = fopen(target, 'w');
if fid == -1
    error('generate_oracles:cannotWrite', 'Could not open %s', target);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(data, 'PrettyPrint', true));
end


function writeProvenance(oracleDir)
% Record which RAVEN produced these answers, so a stale oracle is visible.
commit = 'unknown';
try
    ravenDir = fileparts(which('readYAMLmodel'));
    [status, out] = system(sprintf('git -C "%s" rev-parse HEAD', ravenDir));
    if status == 0
        commit = strtrim(out);
    end
catch
    % Leave as unknown -- provenance is best-effort, not a hard requirement.
end

text = sprintf([ ...
    '# Oracle provenance\n\n' ...
    'Generated by `scripts/parity/generate_oracles.m`.\n\n' ...
    '- RAVEN commit: `%s`\n' ...
    '- MATLAB: %s\n' ...
    '- Generated: %s\n\n' ...
    'Regenerate after changing a fixture or after a RAVEN change that could\n' ...
    'affect these functions, and commit the result.\n'], ...
    commit, version, datestr(now, 'yyyy-mm-dd'));

fid = fopen(fullfile(oracleDir, 'PROVENANCE.md'), 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', text);
end
