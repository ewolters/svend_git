/**
 * Generate the full config form HTML for any analysis type.
 * Ported from the canonical analysis_workbench.html (4,745 lines).
 *
 * @param {string} type - analysis type (stats, spc, bayesian, viz, etc.)
 * @param {string} id - analysis ID (ttest, imr, capability, etc.)
 * @param {Array} columns - [{name, dtype}] column metadata from active dataset
 * @returns {string} HTML for the config form
 */
function generateConfigForm(type, id, columns) {
    columns = columns || [];
    const colOptions = columns.map(c => `<option value="${c.name}">${c.name}</option>`).join('');
    const numCols = columns.filter(c => c.dtype === 'numeric').map(c => `<option value="${c.name}">${c.name}</option>`).join('') || colOptions;

    if (type === 'stats') {
        if (id === 'descriptive') {
            const numericCols = columns.filter(c => c.dtype === 'numeric');
            const checkboxes = numericCols.map(c => `
                <label class="aw-checkbox-item">
                    <input type="checkbox" name="vars" value="${c.name}" checked>
                    <span>${c.name}</span>
                </label>
            `).join('');

            return `
                <div class="aw-form-group">
                    <label>Variables:</label>
                    <div class="aw-checkbox-list" style="max-height:200px;overflow-y:auto;">
                        ${checkboxes || '<span style="color:#7a8f7a;">No numeric columns</span>'}
                    </div>
                    <div style="margin-top:6px;">
                        <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('vars', true)">Select All</button>
                        <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('vars', false)">Clear</button>
                    </div>
                </div>
            `;
        }
        if (id === 'ttest') {
            return `
                <div class="aw-form-group">
                    <label>Sample:</label>
                    <select id="cfg-var1">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Test mean:</label>
                    <input type="number" id="cfg-mu" value="0" step="any">
                </div>
                <div class="aw-form-group">
                    <label>Confidence level:</label>
                    <select id="cfg-conf">
                        <option value="95">95%</option>
                        <option value="99">99%</option>
                        <option value="90">90%</option>
                    </select>
                </div>
            `;
        }
        if (id === 'ttest2') {
            return `
                ${twoSampleLayout(numCols, colOptions, {label1: 'Sample 1', label2: 'Sample 2', uid: 'tt2'})}
                <div class="aw-form-group">
                    <label>Confidence level:</label>
                    <select id="cfg-conf">
                        <option value="95">95%</option>
                        <option value="99">99%</option>
                        <option value="90">90%</option>
                    </select>
                </div>
            `;
        }
        if (id === 'paired_t') {
            return `
                ${twoSampleLayout(numCols, colOptions, {label1: 'Before / Sample 1', label2: 'After / Sample 2', uid: 'pt'})}
                <div class="aw-form-group">
                    <label>Confidence level:</label>
                    <select id="cfg-conf">
                        <option value="95">95%</option>
                        <option value="99">99%</option>
                        <option value="90">90%</option>
                    </select>
                </div>
            `;
        }
        if (id === 'anova') {
            const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
            return `
                <div class="aw-form-group">
                    <label>Response (measurement):</label>
                    <select id="cfg-response">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Factor (groups):</label>
                    <select id="cfg-factor">${catCols || colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Confidence level:</label>
                    <select id="cfg-conf">
                        <option value="95">95%</option>
                        <option value="99">99%</option>
                        <option value="90">90%</option>
                    </select>
                </div>
            `;
        }
        if (['tukey_hsd', 'games_howell', 'dunn', 'scheffe_test', 'bonferroni_test'].includes(id)) {
            const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
            return `
                <div class="aw-form-group">
                    <label>Response (measurement):</label>
                    <select id="cfg-response">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Factor (groups):</label>
                    <select id="cfg-factor">${catCols || colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Confidence level:</label>
                    <select id="cfg-conf">
                        <option value="95">95%</option>
                        <option value="99">99%</option>
                        <option value="90">90%</option>
                    </select>
                </div>
            `;
        }
        if (id === 'dunnett') {
            const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
            return `
                <div class="aw-form-group">
                    <label>Response (measurement):</label>
                    <select id="cfg-response">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Factor (groups):</label>
                    <select id="cfg-factor">${catCols || colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Control group:</label>
                    <select id="cfg-control">${catCols || colOptions}</select>
                    <small style="color:#7a8f7a;">Select factor first, then pick the control level</small>
                </div>
                <div class="aw-form-group">
                    <label>Confidence level:</label>
                    <select id="cfg-conf">
                        <option value="95">95%</option>
                        <option value="99">99%</option>
                        <option value="90">90%</option>
                    </select>
                </div>
            `;
        }
        if (id === 'hsu_mcb') {
            const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
            return `
                <div class="aw-form-group">
                    <label>Response (measurement):</label>
                    <select id="cfg-response">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Factor (groups):</label>
                    <select id="cfg-factor">${catCols || colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Best is:</label>
                    <select id="cfg-direction">
                        <option value="max">Maximum (larger is better)</option>
                        <option value="min">Minimum (smaller is better)</option>
                    </select>
                </div>
                <div class="aw-form-group">
                    <label>Confidence level:</label>
                    <select id="cfg-conf">
                        <option value="95">95%</option>
                        <option value="99">99%</option>
                        <option value="90">90%</option>
                    </select>
                </div>
            `;
        }
        if (['hotelling_t2', 'manova'].includes(id)) {
            const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
            const numericCols = columns.filter(c => c.dtype === 'numeric');
            const checkboxes = numericCols.map(c => `
                <label class="aw-checkbox-item">
                    <input type="checkbox" name="responses" value="${c.name}">
                    <span>${c.name}</span>
                </label>
            `).join('');
            return `
                <div class="aw-form-group">
                    <label>Response variables (select 2+):</label>
                    <div class="aw-checkbox-list" style="max-height:150px;overflow-y:auto;">
                        ${checkboxes || '<span style="color:#7a8f7a;">No numeric columns</span>'}
                    </div>
                    <div style="margin-top:6px;">
                        <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('responses', true)">Select All</button>
                        <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('responses', false)">Clear</button>
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Factor (groups):</label>
                    <select id="cfg-factor">${catCols || colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Confidence level:</label>
                    <select id="cfg-conf">
                        <option value="95">95%</option>
                        <option value="99">99%</option>
                        <option value="90">90%</option>
                    </select>
                </div>
            `;
        }
        if (id === 'anova2') {
            const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
            return `
                <div class="aw-form-group">
                    <label>Response (measurement):</label>
                    <select id="cfg-response">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Factor A:</label>
                    <select id="cfg-factor_a">${catCols || colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Factor B:</label>
                    <select id="cfg-factor_b">${catCols || colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Confidence level:</label>
                    <select id="cfg-conf">
                        <option value="95">95%</option>
                        <option value="99">99%</option>
                        <option value="90">90%</option>
                    </select>
                </div>
            `;
        }
        if (id === 'nested_anova') {
            const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
            return `
                <div class="aw-form-group">
                    <label>Response (measurement):</label>
                    <select id="cfg-response">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Fixed factor (treatment):</label>
                    <select id="cfg-fixed_factor">${catCols || colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Random factor (nesting):</label>
                    <select id="cfg-random_factor">${catCols || colOptions}</select>
                    <small style="color:#7a8f7a;">E.g., operators within machines, batches within suppliers</small>
                </div>
                <div class="aw-form-group">
                    <label>Confidence level:</label>
                    <select id="cfg-conf">
                        <option value="95">95%</option>
                        <option value="99">99%</option>
                        <option value="90">90%</option>
                    </select>
                </div>
            `;
        }
        if (id === 'correlation') {
            const numericCols = columns.filter(c => c.dtype === 'numeric');
            const checkboxes = numericCols.map(c => `
                <label class="aw-checkbox-item">
                    <input type="checkbox" name="vars" value="${c.name}" checked>
                    <span>${c.name}</span>
                </label>
            `).join('');

            return `
                <div class="aw-form-group">
                    <label>Variables:</label>
                    <div class="aw-checkbox-list" style="max-height:150px;overflow-y:auto;">
                        ${checkboxes || '<span style="color:#7a8f7a;">No numeric columns</span>'}
                    </div>
                    <div style="margin-top:6px;">
                        <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('vars', true)">Select All</button>
                        <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('vars', false)">Clear</button>
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Method:</label>
                    <select id="cfg-method">
                        <option value="pearson">Pearson (linear)</option>
                        <option value="spearman">Spearman (rank)</option>
                        <option value="kendall">Kendall (concordance)</option>
                    </select>
                </div>
            `;
        }
        if (id === 'normality') {
            return `
                <div class="aw-form-group">
                    <label>Variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Test:</label>
                    <select id="cfg-test">
                        <option value="anderson">Anderson-Darling</option>
                        <option value="shapiro">Shapiro-Wilk</option>
                        <option value="ks">Kolmogorov-Smirnov</option>
                    </select>
                </div>
            `;
        }
        if (id === 'chi2') {
            return `
                <div class="aw-form-group">
                    <label>Row variable:</label>
                    <select id="cfg-row_var">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Column variable:</label>
                    <select id="cfg-col_var">${colOptions}</select>
                </div>
            `;
        }
        if (id === 'granger') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Potential Cause (X):</label>
                        <select id="cfg-var_x">${numCols}</select>
                    </div>
                    <div class="aw-form-group">
                        <label>Effect (Y):</label>
                        <select id="cfg-var_y">${numCols}</select>
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Maximum Lags:</label>
                    <select id="cfg-max_lag">
                        <option value="2">2</option>
                        <option value="4" selected>4</option>
                        <option value="6">6</option>
                        <option value="8">8</option>
                        <option value="12">12</option>
                    </select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Causal Inference:</strong> Tests if past values of X improve prediction of Y. Significant results suggest X temporally precedes and helps predict Y.
                </div>
            `;
        }
        if (id === 'changepoint') {
            return `
                <div class="aw-form-group">
                    <label>Variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Penalty (sensitivity):</label>
                    <select id="cfg-penalty">
                        <option value="bic" selected>BIC (balanced)</option>
                        <option value="aic">AIC (more sensitive)</option>
                    </select>
                </div>
                <div class="aw-form-group">
                    <label>Minimum segment size:</label>
                    <select id="cfg-min_size">
                        <option value="5">5</option>
                        <option value="10" selected>10</option>
                        <option value="20">20</option>
                        <option value="30">30</option>
                    </select>
                </div>
                <div style="background:rgba(232,87,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e85747;">When Did It Change?</strong> Detects points where the process behavior shifted. Useful for identifying when a cause started affecting the system.
                </div>
            `;
        }
        if (id === 'mann_whitney') {
            return `
                <div class="aw-form-group">
                    <label>Test variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Grouping variable (2 groups):</label>
                    <select id="cfg-group_var">${colOptions}</select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Non-parametric:</strong> Use when data isn't normally distributed. Tests if the two groups have different medians/distributions.
                </div>
            `;
        }
        if (id === 'kruskal') {
            return `
                <div class="aw-form-group">
                    <label>Test variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Grouping variable:</label>
                    <select id="cfg-group_var">${colOptions}</select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Non-parametric ANOVA:</strong> Tests if 3+ groups have different distributions without assuming normality.
                </div>
            `;
        }
        if (id === 'wilcoxon') {
            return `
                ${twoSampleLayout(numCols, colOptions, {label1: 'Before / Sample 1', label2: 'After / Sample 2', uid: 'wlx'})}
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Non-parametric paired test:</strong> Use when paired differences aren't normally distributed. Tests if the median difference is zero.
                </div>
            `;
        }
        if (id === 'friedman') {
            const numColCheckboxes = columns.filter(c => c.dtype === 'numeric').map(c => `
                <label class="aw-checkbox-item">
                    <input type="checkbox" name="vars" value="${c.name}">
                    <span>${c.name}</span>
                </label>
            `).join('') || '<span style="color:var(--aw-text-muted)">No numeric columns found</span>';

            return `
                <div class="aw-form-group">
                    <label>Repeated measures (select 3+):</label>
                    <div class="aw-checkbox-list" id="cfg-vars-list">
                        ${numColCheckboxes}
                    </div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Non-parametric repeated measures:</strong> Select 3+ measurement columns (e.g., Time1, Time2, Time3). Each row is one subject measured under all conditions.
                </div>
            `;
        }
        if (id === 'spearman') {
            return `
                <div class="aw-form-group">
                    <label>Variable X:</label>
                    <select id="cfg-var1">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Variable Y:</label>
                    <select id="cfg-var2">${numCols}</select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Rank correlation:</strong> Measures monotonic association without assuming linearity. Returns rho, p-value, and confidence interval.
                </div>
            `;
        }
        if (id === 'main_effects') {
            const catCols = columns.filter(c => c.dtype === 'text').map(c => c.name);
            const factorCheckboxes = catCols.map(c => `
                <label class="aw-checkbox-item">
                    <input type="checkbox" name="factors" value="${c}">
                    <span>${c}</span>
                </label>
            `).join('') || '<span style="color:var(--aw-text-muted)">No categorical columns found</span>';

            return `
                <div class="aw-form-group">
                    <label>Response (Y):</label>
                    <select id="cfg-response">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Factors:</label>
                    <div class="aw-checkbox-list" id="cfg-factors-list">
                        ${factorCheckboxes}
                    </div>
                </div>
                <div style="background:rgba(232,149,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e89547;">DOE Main Effects:</strong> Shows the average effect of each factor level on the response. Parallel lines suggest no interaction; non-parallel lines hint at interactions.
                </div>
            `;
        }
        if (id === 'interaction') {
            const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
            const catOptions = catCols || colOptions;

            return `
                <div class="aw-form-group">
                    <label>Response (Y):</label>
                    <select id="cfg-response">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Factor 1 (X axis):</label>
                        <select id="cfg-factor1">${catOptions}</select>
                    </div>
                    <div class="aw-form-group">
                        <label>Factor 2 (Lines):</label>
                        <select id="cfg-factor2">${catOptions}</select>
                    </div>
                </div>
                <div style="background:rgba(232,149,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e89547;">Interaction Plot:</strong> If lines are parallel, no interaction exists. Crossing or diverging lines indicate the effect of Factor 1 depends on Factor 2.
                </div>
            `;
        }
        if (id === 'logistic') {
            const numericCols = columns.filter(c => c.dtype === 'numeric');
            const checkboxes = numericCols.map(c => `
                <label class="aw-checkbox-item">
                    <input type="checkbox" name="predictors" value="${c.name}">
                    <span>${c.name}</span>
                </label>
            `).join('');

            return `
                <div class="aw-form-group">
                    <label>Response (binary Y):</label>
                    <select id="cfg-response">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Predictors:</label>
                    <div class="aw-checkbox-list" id="cfg-predictors-list">
                        ${checkboxes}
                    </div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Logistic Regression:</strong> For binary outcomes. Returns odds ratios (OR > 1 increases probability) and ROC curve.
                </div>
            `;
        }
        if (id === 'f_test') {
            return `
                <div class="aw-form-group">
                    <label>Test variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Grouping variable (2 groups):</label>
                    <select id="cfg-group_var">${colOptions}</select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">F-Test:</strong> Tests if two groups have equal variances. Use before t-tests to check homogeneity assumption.
                </div>
            `;
        }
        if (id === 'equivalence') {
            return `
                <div class="aw-form-group">
                    <label>Test variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Grouping variable (2 groups):</label>
                    <select id="cfg-group_var">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Equivalence margin (±):</label>
                    <input type="number" id="cfg-margin" value="0.5" step="0.1" min="0">
                </div>
                <div style="background:rgba(232,149,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e89547;">TOST:</strong> Tests if groups are equivalent within margin. Different from t-test (absence of difference ≠ equivalence).
                </div>
            `;
        }
        if (id === 'runs_test') {
            return `
                <div class="aw-form-group">
                    <label>Variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Runs Test:</strong> Tests if sequence is random. Too few runs = clustering/trends. Too many = oscillation.
                </div>
            `;
        }
        if (id === 'multi_vari') {
            const catCols = columns.filter(c => c.dtype === 'text').map(c => c.name);
            const factorCheckboxes = catCols.map(c => `
                <label class="aw-checkbox-item">
                    <input type="checkbox" name="factors" value="${c}">
                    <span>${c}</span>
                </label>
            `).join('') || '<span style="color:var(--aw-text-muted)">No categorical columns found</span>';

            return `
                <div class="aw-form-group">
                    <label>Response (Y):</label>
                    <select id="cfg-response">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Factors (select 1-3):</label>
                    <div class="aw-checkbox-list" id="cfg-factors-list">
                        ${factorCheckboxes}
                    </div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Multi-Vari:</strong> Shows within-group vs between-group variation. Compare spreads to identify dominant sources.
                </div>
            `;
        }
        if (id === 'regression') {
            const numericCols = columns.filter(c => c.dtype === 'numeric');
            const checkboxes = numericCols.map(c => `
                <label class="aw-checkbox-item">
                    <input type="checkbox" name="predictors" value="${c.name}">
                    <span>${c.name}</span>
                </label>
            `).join('');

            return `
                <div class="aw-form-group">
                    <label>Response (Y):</label>
                    <select id="cfg-response">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Predictors (X):</label>
                    <div class="aw-checkbox-list" id="cfg-predictors-list">
                        ${checkboxes}
                    </div>
                    <div style="margin-top:6px;">
                        <button type="button" class="aw-btn-small" onclick="selectAllPredictors(true)">Select All</button>
                        <button type="button" class="aw-btn-small" onclick="selectAllPredictors(false)">Clear</button>
                    </div>
                </div>
                <div class="aw-form-row" style="display:flex;gap:12px;">
                    <div class="aw-form-group" style="flex:1;">
                        <label>Model type:</label>
                        <select id="cfg-degree">
                            <option value="1">Linear</option>
                            <option value="2">Quadratic (X²)</option>
                            <option value="3">Cubic (X³)</option>
                        </select>
                    </div>
                    <div class="aw-form-group" style="flex:1;">
                        <label>Interactions:</label>
                        <select id="cfg-interactions">
                            <option value="none">None</option>
                            <option value="all">All pairs (X₁·X₂)</option>
                        </select>
                    </div>
                </div>
            `;
        }
    }

    if (type === 'spc') {
        // Historical limits fields (shared by I-MR and X-bar R/S)
        const historicalFields = `
                <details style="margin-top:8px;">
                    <summary style="cursor:pointer;font-size:11px;color:#9aaa9a;">Phase 2: Historical limits</summary>
                    <div style="padding:8px 0 0 12px;">
                        <div class="aw-form-group">
                            <label>Historical mean (leave blank to calculate):</label>
                            <input type="number" id="cfg-historical_mean" step="any" placeholder="Auto">
                        </div>
                        <div class="aw-form-group">
                            <label>Historical sigma (leave blank to estimate):</label>
                            <input type="number" id="cfg-historical_sigma" step="any" min="0" placeholder="Auto">
                        </div>
                    </div>
                </details>
                <div id="spc-gage-info" style="margin-top:8px;font-size:10px;color:#9aaa9a;"></div>`;
        // Load recent gage studies for MSA context
        setTimeout(() => {
            const infoEl = document.getElementById('spc-gage-info');
            if (!infoEl) return;
            fetch('/api/dsw/measurement-systems/recent/', {
                headers: { 'X-CSRFToken': getCSRFToken() }
            }).then(r => r.json()).then(d => {
                const systems = d.measurement_systems || [];
                if (systems.length === 0) return;
                let html = '<details><summary style="cursor:pointer;">Measurement systems (' + systems.length + ')</summary><div style="padding:4px 0 0 8px;">';
                systems.forEach(s => {
                    const color = s.assessment === 'Acceptable' ? '#4a9f6e' : s.assessment === 'Marginal' ? '#f59e0b' : '#e74c3c';
                    html += '<div style="margin-bottom:4px;"><span style="color:' + color + ';">●</span> ' + s.system_name + ' — ' + s.grr_percent + '% GRR (' + s.assessment + ')';
                    if (s.ndc) html += ', NDC=' + s.ndc;
                    html += '</div>';
                });
                html += '</div></details>';
                infoEl.innerHTML = html;
            }).catch(() => {});
        }, 100);
        if (id === 'imr') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                ${historicalFields}
            `;
        }
        if (id === 'xbar_r' || id === 'xbar_s') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Subgroup column:</label>
                    <select id="cfg-subgroup">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Subgroup size:</label>
                    <input type="number" id="cfg-subgroup_size" value="5" min="2" max="25">
                </div>
                ${historicalFields}
            `;
        }
        if (id === 'p_chart') {
            return `
                <div class="aw-form-group">
                    <label>Defectives column:</label>
                    <select id="cfg-defectives">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Sample size column:</label>
                    <select id="cfg-sample_size">${numCols}</select>
                </div>
            `;
        }
        if (id === 'capability') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Lower spec (LSL):</label>
                        <input type="number" id="cfg-lsl" step="any">
                    </div>
                    <div class="aw-form-group">
                        <label>Upper spec (USL):</label>
                        <input type="number" id="cfg-usl" step="any">
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Target (optional):</label>
                    <input type="number" id="cfg-target" step="any">
                </div>
            `;
        }
        if (id === 'np_chart') {
            return `
                <div class="aw-form-group">
                    <label>Defectives column:</label>
                    <select id="cfg-defectives">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Sample size (constant):</label>
                    <input type="number" id="cfg-sample_size" value="50" min="1">
                </div>
            `;
        }
        if (id === 'c_chart') {
            return `
                <div class="aw-form-group">
                    <label>Defects column:</label>
                    <select id="cfg-defects">${numCols}</select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">C Chart:</strong> For count of defects per unit when inspection unit is constant.
                </div>
            `;
        }
        if (id === 'u_chart') {
            return `
                <div class="aw-form-group">
                    <label>Defects column:</label>
                    <select id="cfg-defects">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Units inspected column:</label>
                    <select id="cfg-units">${numCols}</select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">U Chart:</strong> Defects per unit when sample size varies.
                </div>
            `;
        }
        if (id === 'cusum') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Target value (0 = use mean):</label>
                    <input type="number" id="cfg-target" value="0" step="any">
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>k (slack):</label>
                        <input type="number" id="cfg-k" value="0.5" step="0.1" min="0">
                    </div>
                    <div class="aw-form-group">
                        <label>h (decision):</label>
                        <input type="number" id="cfg-h" value="5" step="0.5" min="1">
                    </div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">CUSUM:</strong> Detects small sustained shifts. More sensitive than Shewhart charts.
                </div>
            `;
        }
        if (id === 'ewma') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Target value (0 = use mean):</label>
                    <input type="number" id="cfg-target" value="0" step="any">
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>λ (smoothing 0-1):</label>
                        <input type="number" id="cfg-lambda" value="0.2" step="0.05" min="0.05" max="1">
                    </div>
                    <div class="aw-form-group">
                        <label>L (sigma width):</label>
                        <input type="number" id="cfg-L" value="3" step="0.5" min="1">
                    </div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">EWMA:</strong> Good for detecting small shifts. Lower λ = more smoothing.
                </div>
            `;
        }
        if (id === 'nonnormal_capability') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>LSL:</label><input type="number" id="cfg-lsl" step="any"></div>
                    <div class="aw-form-group"><label>USL:</label><input type="number" id="cfg-usl" step="any"></div>
                </div>
                <div class="aw-form-group">
                    <label>Target (optional):</label>
                    <input type="number" id="cfg-target" step="any" placeholder="Midpoint if blank">
                </div>
                <div style="background:rgba(232,149,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e89547;">Non-Normal Capability:</strong> Fits best distribution (Weibull, lognormal, etc.) then computes Ppk/Pp. Use when data fails normality test.
                </div>
            `;
        }
        if (id === 'degradation_capability') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Time column:</label>
                    <select id="cfg-time_column">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>LSL:</label><input type="number" id="cfg-lsl" step="any"></div>
                    <div class="aw-form-group"><label>USL:</label><input type="number" id="cfg-usl" step="any"></div>
                </div>
                <div class="aw-form-group">
                    <label>Target Cpk:</label>
                    <input type="number" id="cfg-target_cpk" value="1.33" step="0.01" min="0">
                </div>
                <div style="background:rgba(217,74,74,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#d94a4a;">Degradation Capability:</strong> Tracks capability over time windows. Detects when process drifts toward spec limits.
                </div>
            `;
        }
        if (id === 'between_within') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Subgroup column:</label>
                    <select id="cfg-subgroup">${colOptions}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>LSL:</label><input type="number" id="cfg-lsl" step="any"></div>
                    <div class="aw-form-group"><label>USL:</label><input type="number" id="cfg-usl" step="any"></div>
                </div>
                <div style="background:rgba(74,144,217,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#4a90d9;">Between/Within:</strong> Separates between-subgroup and within-subgroup variation. Ppk uses total, Cpk uses within.
                </div>
            `;
        }
        if (id === 'laney_p') {
            return `
                <div class="aw-form-group">
                    <label>Defectives column:</label>
                    <select id="cfg-defectives">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Sample size column:</label>
                    <select id="cfg-sample_size">${numCols}</select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Laney P':</strong> Adjusted P chart for overdispersed data. Prevents false alarms when large sample sizes inflate sensitivity.
                </div>
            `;
        }
        if (id === 'laney_u') {
            return `
                <div class="aw-form-group">
                    <label>Defects column:</label>
                    <select id="cfg-defects">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Units column:</label>
                    <select id="cfg-units">${numCols}</select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Laney U':</strong> Adjusted U chart for overdispersed defect rates. Corrects control limits when data variance exceeds Poisson assumption.
                </div>
            `;
        }
        if (id === 'moving_average') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Span (window size):</label>
                    <input type="number" id="cfg-span" value="5" min="2" max="50">
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Moving Average:</strong> Smooths individual observations with a rolling window. Good for autocorrelated data.
                </div>
            `;
        }
        if (id === 'zone_chart') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div style="background:rgba(74,144,217,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#4a90d9;">Zone Chart:</strong> Scores each point by zone (A=4, B=2, C=0). Out of control when cumulative score ≥ 8.
                </div>
            `;
        }
        if (id === 'mewma') {
            const mvCheckboxes = columns.filter(c => c.dtype === 'numeric').map(c =>
                `<label class="aw-checkbox-item"><input type="checkbox" name="variables" value="${c.name}" checked><span>${c.name}</span></label>`
            ).join('');
            return `
                <div class="aw-form-group">
                    <label>Variables:</label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">${mvCheckboxes}</div>
                </div>
                <div class="aw-form-group">
                    <label>λ (smoothing 0-1):</label>
                    <input type="number" id="cfg-lambda" value="0.2" step="0.05" min="0.05" max="1">
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">MEWMA:</strong> Multivariate EWMA for monitoring correlated variables simultaneously. Detects small shifts in the mean vector.
                </div>
            `;
        }
        if (id === 'generalized_variance') {
            const mvCheckboxes = columns.filter(c => c.dtype === 'numeric').map(c =>
                `<label class="aw-checkbox-item"><input type="checkbox" name="variables" value="${c.name}" checked><span>${c.name}</span></label>`
            ).join('');
            return `
                <div class="aw-form-group">
                    <label>Variables:</label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">${mvCheckboxes}</div>
                </div>
                <div class="aw-form-group">
                    <label>Subgroup column (optional):</label>
                    <select id="cfg-subgroup"><option value="">None</option>${colOptions}</select>
                </div>
                <div style="background:rgba(232,149,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e89547;">Generalized Variance:</strong> Monitors the determinant of the covariance matrix. Detects changes in multivariate dispersion.
                </div>
            `;
        }
        if (id === 'entropy_spc') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Alpha:</label>
                        <input type="number" id="cfg-alpha" value="0.05" step="0.01" min="0.001" max="0.5">
                    </div>
                    <div class="aw-form-group">
                        <label>Window size:</label>
                        <input type="number" id="cfg-window" value="20" min="5" max="200">
                    </div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Entropy SPC:</strong> Information-theoretic control chart. Detects distributional changes beyond just mean/variance shifts.
                </div>
            `;
        }
        return `<p style="color:#9aaa9a;font-size:11px;">Select SPC chart type.</p>`;
    }

    if (type === 'timeseries') {
        if (id === 'arima') {
            return `
                <div class="aw-form-group">
                    <label>Variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>p (AR order):</label>
                        <select id="cfg-p">
                            <option value="0">0</option>
                            <option value="1" selected>1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                        </select>
                    </div>
                    <div class="aw-form-group">
                        <label>d (differencing):</label>
                        <select id="cfg-d">
                            <option value="0">0</option>
                            <option value="1" selected>1</option>
                            <option value="2">2</option>
                        </select>
                    </div>
                    <div class="aw-form-group">
                        <label>q (MA order):</label>
                        <select id="cfg-q">
                            <option value="0">0</option>
                            <option value="1" selected>1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                        </select>
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Forecast periods:</label>
                    <input type="number" id="cfg-forecast" value="10" min="1" max="100">
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">ARIMA:</strong> Use ACF/PACF to determine orders. d=1 for trending data.
                </div>
            `;
        }
        if (id === 'sarima') {
            return `
                <div class="aw-form-group">
                    <label>Variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>p (AR):</label>
                        <select id="cfg-p"><option value="0">0</option><option value="1" selected>1</option><option value="2">2</option><option value="3">3</option></select></div>
                    <div class="aw-form-group"><label>d (diff):</label>
                        <select id="cfg-d"><option value="0">0</option><option value="1" selected>1</option><option value="2">2</option></select></div>
                    <div class="aw-form-group"><label>q (MA):</label>
                        <select id="cfg-q"><option value="0">0</option><option value="1" selected>1</option><option value="2">2</option><option value="3">3</option></select></div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>P (Seasonal AR):</label>
                        <select id="cfg-P"><option value="0">0</option><option value="1" selected>1</option><option value="2">2</option></select></div>
                    <div class="aw-form-group"><label>D (Seasonal diff):</label>
                        <select id="cfg-D"><option value="0">0</option><option value="1" selected>1</option></select></div>
                    <div class="aw-form-group"><label>Q (Seasonal MA):</label>
                        <select id="cfg-Q"><option value="0">0</option><option value="1" selected>1</option><option value="2">2</option></select></div>
                </div>
                <div class="aw-form-group">
                    <label>Seasonal period (m):</label>
                    <select id="cfg-m">
                        <option value="4">4 (Quarterly)</option>
                        <option value="7">7 (Weekly)</option>
                        <option value="12" selected>12 (Monthly)</option>
                        <option value="24">24 (Bi-monthly)</option>
                        <option value="52">52 (Weekly/Yearly)</option>
                    </select>
                </div>
                <div class="aw-form-group">
                    <label>Forecast periods:</label>
                    <input type="number" id="cfg-forecast" value="24" min="1" max="200">
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">SARIMA:</strong> (p,d,q) = non-seasonal, (P,D,Q)[m] = seasonal. Use Decomposition to identify seasonal period.
                </div>
            `;
        }
        if (id === 'decomposition') {
            return `
                <div class="aw-form-group">
                    <label>Variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Seasonal period:</label>
                        <select id="cfg-period">
                            <option value="4">4 (Quarterly)</option>
                            <option value="7">7 (Weekly)</option>
                            <option value="12" selected>12 (Monthly)</option>
                            <option value="24">24 (Hourly)</option>
                            <option value="52">52 (Yearly weeks)</option>
                        </select>
                    </div>
                    <div class="aw-form-group">
                        <label>Model:</label>
                        <select id="cfg-model">
                            <option value="additive" selected>Additive</option>
                            <option value="multiplicative">Multiplicative</option>
                        </select>
                    </div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Decomposition:</strong> Separates trend, seasonal, and residual components.
                </div>
            `;
        }
        if (id === 'acf_pacf') {
            return `
                <div class="aw-form-group">
                    <label>Variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Maximum lags:</label>
                    <input type="number" id="cfg-lags" value="20" min="5" max="100">
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">ACF/PACF:</strong> Use to identify ARIMA orders. PACF cuts off → AR(p), ACF cuts off → MA(q).
                </div>
            `;
        }
        return `<p style="color:#9aaa9a;font-size:11px;">Select time series analysis.</p>`;
    }

    if (type === 'reliability') {
        if (id === 'weibull') {
            return `
                <div class="aw-form-group">
                    <label>Time to failure:</label>
                    <select id="cfg-time">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Censoring column (optional, 1=failed):</label>
                    <select id="cfg-censor"><option value="">All failed</option>${numCols}</select>
                </div>
                <div style="background:rgba(232,149,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e89547;">Weibull:</strong> β<1 = infant mortality, β=1 = random, β>1 = wear-out failures.
                </div>
            `;
        }
        if (id === 'kaplan_meier') {
            return `
                <div class="aw-form-group">
                    <label>Time variable:</label>
                    <select id="cfg-time">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Event indicator (1=event, 0=censored):</label>
                    <select id="cfg-event">
                        <option value="">All events (no censoring)</option>
                        ${numCols}
                    </select>
                </div>
                <div class="aw-form-group">
                    <label>Group by (optional):</label>
                    <select id="cfg-group">
                        <option value="">No grouping</option>
                        ${colOptions}
                    </select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Kaplan-Meier:</strong> Survival curves. Compare groups to see if survival differs.
                </div>
            `;
        }
        if (id === 'lognormal' || id === 'exponential') {
            const distName = id === 'lognormal' ? 'Lognormal' : 'Exponential';
            const distNote = id === 'lognormal'
                ? 'Lognormal fits right-skewed failure data. Common for fatigue, corrosion, and chemical degradation.'
                : 'Exponential assumes constant hazard rate (memoryless). Use for electronic components or random failures.';
            return `
                <div class="aw-form-group">
                    <label>Time to failure:</label>
                    <select id="cfg-time">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Censoring column (optional, 1=failed):</label>
                    <select id="cfg-censor"><option value="">All failed</option>${numCols}</select>
                </div>
                <div style="background:rgba(232,149,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e89547;">${distName}:</strong> ${distNote}
                </div>
            `;
        }
        if (id === 'distribution_id') {
            return `
                <div class="aw-form-group">
                    <label>Time to failure:</label>
                    <select id="cfg-time">${numCols}</select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Distribution ID:</strong> Fits Weibull, lognormal, exponential, and normal. Compares AIC/BIC to recommend the best model.
                </div>
            `;
        }
        if (id === 'reliability_test_plan') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Target reliability:</label>
                        <input type="number" id="cfg-target_reliability" value="0.90" step="0.01" min="0.5" max="0.999">
                    </div>
                    <div class="aw-form-group">
                        <label>Confidence:</label>
                        <input type="number" id="cfg-confidence" value="0.95" step="0.01" min="0.5" max="0.999">
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Test duration:</label>
                        <input type="number" id="cfg-test_duration" value="1000" min="1">
                    </div>
                    <div class="aw-form-group">
                        <label>Distribution:</label>
                        <select id="cfg-distribution">
                            <option value="exponential">Exponential</option>
                            <option value="weibull">Weibull</option>
                        </select>
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Weibull β (shape, if applicable):</label>
                    <input type="number" id="cfg-beta" value="1.0" step="0.1" min="0.1">
                </div>
                <div style="background:rgba(74,144,217,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#4a90d9;">Test Plan:</strong> Calculates sample size needed to demonstrate reliability target at given confidence. No dataset required.
                </div>
            `;
        }
        if (id === 'accelerated_life') {
            return `
                <div class="aw-form-group">
                    <label>Time to failure:</label>
                    <select id="cfg-time">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Stress variable:</label>
                    <select id="cfg-stress">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Model:</label>
                        <select id="cfg-model">
                            <option value="arrhenius">Arrhenius (temperature)</option>
                            <option value="inverse_power">Inverse Power (voltage/load)</option>
                            <option value="eyring">Eyring</option>
                        </select>
                    </div>
                    <div class="aw-form-group">
                        <label>Use stress level:</label>
                        <input type="number" id="cfg-use_stress" step="any" placeholder="Normal operating">
                    </div>
                </div>
                <div style="background:rgba(217,74,74,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#d94a4a;">Accelerated Life:</strong> Extrapolates failure times from high-stress tests to normal conditions. Arrhenius for temperature, Inverse Power for voltage/mechanical stress.
                </div>
            `;
        }
        if (id === 'repairable_systems') {
            return `
                <div class="aw-form-group">
                    <label>Failure time column:</label>
                    <select id="cfg-time">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>System ID column (optional):</label>
                    <select id="cfg-system"><option value="">Single system</option>${colOptions}</select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Repairable Systems:</strong> Tests for trend (improving/deteriorating) using Crow-AMSAA model. ROCOF = Rate of Occurrence of Failures.
                </div>
            `;
        }
        if (id === 'warranty') {
            return `
                <div class="aw-form-group">
                    <label>Time to claim:</label>
                    <select id="cfg-time">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Warranty period:</label>
                        <input type="number" id="cfg-warranty_period" value="12" min="1">
                    </div>
                    <div class="aw-form-group">
                        <label>Fleet size:</label>
                        <input type="number" id="cfg-fleet_size" value="1000" min="1">
                    </div>
                </div>
                <div style="background:rgba(232,149,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e89547;">Warranty:</strong> Projects future warranty claims using current failure data. Accounts for IBNR (Incurred But Not Reported) claims.
                </div>
            `;
        }
        if (id === 'competing_risks') {
            return `
                <div class="aw-form-group">
                    <label>Time to event:</label>
                    <select id="cfg-time">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Event indicator (1=event):</label>
                    <select id="cfg-event"><option value="">All observed</option>${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Failure mode column:</label>
                    <select id="cfg-failure_mode">${colOptions}</select>
                </div>
                <div style="background:rgba(217,74,74,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#d94a4a;">Competing Risks:</strong> Separates multiple failure modes. Cumulative incidence functions show probability of each failure type over time.
                </div>
            `;
        }
        return `<p style="color:#9aaa9a;font-size:11px;">Select reliability analysis.</p>`;
    }

    if (type === 'msa') {
        if (id === 'gage_rr') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Part ID column:</label>
                    <select id="cfg-part">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Operator column:</label>
                    <select id="cfg-operator">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Tolerance (USL \u2212 LSL):</label>
                    <input type="number" id="cfg-tolerance" value="" step="any" min="0" placeholder="Optional">
                </div>
                <div class="aw-form-group" style="display:flex;align-items:center;gap:8px;">
                    <input type="checkbox" id="cfg-compare_bayesian" style="width:auto;">
                    <label for="cfg-compare_bayesian" style="margin:0;cursor:pointer;">Compare with Bayesian Gage R&R</label>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Gage R&R (Crossed):</strong> &lt;10% excellent, 10-30% marginal, &gt;30% unacceptable.
                    %Tolerance = 6\u03C3<sub>GRR</sub> / tolerance \u00D7 100.
                    Bayesian comparison adds posterior distributions and probability-based verdicts.
                </div>
            `;
        }
        if (id === 'gage_rr_nested') {
            return `
                <div class="aw-form-group"><label>Measurement column:</label><select id="cfg-measurement">${numCols}</select></div>
                <div class="aw-form-group"><label>Part ID column:</label><select id="cfg-part">${colOptions}</select></div>
                <div class="aw-form-group"><label>Operator column:</label><select id="cfg-operator">${colOptions}</select></div>
                <div style="background:rgba(74,144,217,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#4a90d9;">Nested Gage R&R:</strong> For destructive testing where operators measure different parts.
                </div>
            `;
        }
        if (id === 'gage_linearity_bias') {
            return `
                <div class="aw-form-group"><label>Measurement column:</label><select id="cfg-measurement">${numCols}</select></div>
                <div class="aw-form-group"><label>Reference column:</label><select id="cfg-reference">${numCols}</select></div>
                <div style="background:rgba(232,149,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e89547;">Linearity & Bias:</strong> Tests if bias changes across the measurement range. Slope ≠ 0 indicates linearity problem.
                </div>
            `;
        }
        if (id === 'gage_type1') {
            return `
                <div class="aw-form-group"><label>Measurement column:</label><select id="cfg-measurement">${numCols}</select></div>
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>Reference value:</label><input type="number" id="cfg-reference" value="0" step="any"></div>
                    <div class="aw-form-group"><label>Tolerance (USL − LSL):</label><input type="number" id="cfg-tolerance" value="1.0" step="any" min="0.001"></div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Type 1 Gage Study:</strong> One part, one operator, repeated measurements. Cg ≥ 1.33 and Cgk ≥ 1.33 required.
                </div>
            `;
        }
        if (id === 'attribute_gage') {
            return `
                <div class="aw-form-group"><label>Appraiser result column (pass/fail):</label><select id="cfg-result">${colOptions}</select></div>
                <div class="aw-form-group"><label>Reference (truth) column:</label><select id="cfg-reference">${colOptions}</select></div>
                <div class="aw-form-group"><label>Appraiser column (optional):</label><select id="cfg-appraiser"><option value="">(none)</option>${colOptions}</select></div>
                <div style="background:rgba(217,74,74,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#d94a4a;">Attribute Gage:</strong> Compares pass/fail calls against known reference. ≥90% agreement required.
                </div>
            `;
        }
        if (id === 'attribute_agreement') {
            return `
                <div class="aw-form-group"><label>Appraiser column:</label><select id="cfg-appraiser">${colOptions}</select></div>
                <div class="aw-form-group"><label>Part/Item column:</label><select id="cfg-part">${colOptions}</select></div>
                <div class="aw-form-group"><label>Rating column:</label><select id="cfg-rating">${colOptions}</select></div>
                <div style="background:rgba(74,144,217,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#4a90d9;">Attribute Agreement:</strong> Cohen's κ (2 raters) or Fleiss' κ (3+). κ > 0.6 = substantial agreement.
                </div>
            `;
        }
        return `<p style="color:#9aaa9a;font-size:11px;">Select MSA analysis.</p>`;
    }

    if (type === 'ml') {
        const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
        const numericCols = columns.filter(c => c.dtype === 'numeric');
        const featureCheckboxes = columns.map(c => `
            <label class="aw-checkbox-item">
                <input type="checkbox" name="features" value="${c.name}" ${c.dtype === 'numeric' ? 'checked' : ''}>
                <span>${c.name}</span>
                <span class="aw-var-badge">${c.dtype === 'numeric' ? 'num' : 'cat'}</span>
            </label>
        `).join('');
        const numericFeatureCheckboxes = numericCols.map(c => `
            <label class="aw-checkbox-item">
                <input type="checkbox" name="features" value="${c.name}" checked>
                <span>${c.name}</span>
            </label>
        `).join('');

        if (id === 'classification') {
            return `
                <div class="aw-form-group">
                    <label>Target (class to predict):</label>
                    <select id="cfg-target">${catCols || colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Features:
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">
                        ${featureCheckboxes}
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Algorithm:</label>
                    <select id="cfg-algorithm">
                        <option value="random_forest">Random Forest</option>
                        <option value="xgboost">XGBoost</option>
                        <option value="logistic">Logistic Regression</option>
                        <option value="svm">SVM</option>
                    </select>
                </div>
                <div class="aw-form-group">
                    <label>Test split:</label>
                    <select id="cfg-split">
                        <option value="0.2">20%</option>
                        <option value="0.3">30%</option>
                        <option value="0.25">25%</option>
                    </select>
                </div>
            `;
        }
        if (id === 'clustering') {
            return `
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Features:
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">
                        ${numericFeatureCheckboxes}
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Algorithm:</label>
                    <select id="cfg-algorithm">
                        <option value="kmeans">K-Means</option>
                        <option value="dbscan">DBSCAN</option>
                        <option value="hierarchical">Hierarchical</option>
                    </select>
                </div>
                <div class="aw-form-group">
                    <label>Number of clusters (K-Means):</label>
                    <input type="number" id="cfg-n_clusters" value="3" min="2" max="20">
                </div>
            `;
        }
        if (id === 'pca') {
            return `
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Features:
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">
                        ${numericFeatureCheckboxes}
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Number of components:</label>
                    <input type="number" id="cfg-n_components" value="2" min="1" max="10">
                </div>
                <div class="aw-form-group">
                    <label>Color by (optional):</label>
                    <select id="cfg-color"><option value="">None</option>${colOptions}</select>
                </div>
            `;
        }
        if (id === 'feature') {
            return `
                <div class="aw-form-group">
                    <label>Target:</label>
                    <select id="cfg-target">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Features:
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">
                        ${featureCheckboxes}
                    </div>
                </div>
            `;
        }
        // Phase 1: Causal Lens Toolkit
        if (id === 'bayesian_regression') {
            return `
                <div class="aw-form-group">
                    <label>Target (response):</label>
                    <select id="cfg-target">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Features (predictors):
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">
                        ${numericFeatureCheckboxes}
                    </div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Synara Integration:</strong> Produces coefficient posteriors with 95% credible intervals that feed directly into edge weights.
                </div>
            `;
        }
        if (id === 'gam') {
            return `
                <div class="aw-form-group">
                    <label>Target (response):</label>
                    <select id="cfg-target">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Features:
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">
                        ${numericFeatureCheckboxes}
                    </div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Interpretable Nonlinearity:</strong> Shows smooth spline curves revealing HOW each feature affects the response.
                </div>
            `;
        }
        if (id === 'isolation_forest') {
            return `
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Features:
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">
                        ${numericFeatureCheckboxes}
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Expected contamination (anomaly %):</label>
                    <select id="cfg-contamination">
                        <option value="0.01">1%</option>
                        <option value="0.05" selected>5%</option>
                        <option value="0.10">10%</option>
                        <option value="0.15">15%</option>
                    </select>
                </div>
                <div style="background:rgba(232,87,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e85747;">Missing Cause Signal:</strong> Anomalies are observations that don't fit the model—triggers causal expansion in Synara.
                </div>
            `;
        }
        if (id === 'gaussian_process') {
            return `
                <div class="aw-form-group">
                    <label>Target (Y):</label>
                    <select id="cfg-target">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Predictors:
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">
                        ${numericFeatureCheckboxes}
                    </div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Full Uncertainty:</strong> GP provides confidence bands showing where predictions are reliable vs uncertain.
                </div>
            `;
        }
        if (id === 'pls') {
            return `
                <div class="aw-form-group">
                    <label>Target (Y):</label>
                    <select id="cfg-target">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Predictors:
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">
                        ${numericFeatureCheckboxes}
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Components:</label>
                    <select id="cfg-n_components">
                        <option value="2">2</option>
                        <option value="3" selected>3</option>
                        <option value="4">4</option>
                        <option value="5">5</option>
                    </select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Collinearity Handler:</strong> PLS projects to latent space, avoiding multicollinearity issues common in process data.
                </div>
            `;
        }
        if (id === 'regularized_regression') {
            const predChecks = columns.filter(c => c.dtype === 'numeric').map(c => `
                <label class="aw-checkbox-item"><input type="checkbox" name="predictors" value="${c.name}"><span>${c.name}</span></label>
            `).join('');
            return `
                <div class="aw-form-group">
                    <label>Response:</label>
                    <select id="cfg-response">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Predictors:</label>
                    <div class="aw-checkbox-list" style="max-height:150px;overflow-y:auto;">
                        ${predChecks || '<span style="color:#7a8f7a;">No numeric columns</span>'}
                    </div>
                    <div style="margin-top:6px;">
                        <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('predictors', true)">All</button>
                        <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('predictors', false)">Clear</button>
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Method:</label>
                    <select id="cfg-method">
                        <option value="elastic_net">Elastic Net (Ridge + LASSO)</option>
                        <option value="lasso">LASSO (feature selection)</option>
                        <option value="ridge">Ridge (shrinkage only)</option>
                    </select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Regularized Regression:</strong> Ridge shrinks coefficients, LASSO drops them to zero (feature selection), Elastic Net does both. Alpha selected by cross-validation.
                </div>
            `;
        }
        if (id === 'kaplan_meier') {
            const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
            return `
                <div class="aw-form-group">
                    <label>Time variable (time to event):</label>
                    <select id="cfg-time_col">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Event indicator (1=event, 0=censored):</label>
                    <select id="cfg-event_col">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Group variable (optional):</label>
                    <select id="cfg-group_col"><option value="">None (single curve)</option>${catCols || colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Confidence level:</label>
                    <select id="cfg-conf">
                        <option value="95">95%</option>
                        <option value="99">99%</option>
                        <option value="90">90%</option>
                    </select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Kaplan-Meier:</strong> Non-parametric survival curve. Event indicator: 1 = failure/death, 0 = censored/survived. Group variable enables log-rank test.
                </div>
            `;
        }
        if (id === 'cox_ph') {
            const predChecks = columns.filter(c => c.dtype === 'numeric' || c.dtype === 'text').map(c => `
                <label class="aw-checkbox-item"><input type="checkbox" name="covariates" value="${c.name}"><span>${c.name}</span></label>
            `).join('');
            return `
                <div class="aw-form-group">
                    <label>Time variable (time to event):</label>
                    <select id="cfg-time_col">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Event indicator (1=event, 0=censored):</label>
                    <select id="cfg-event_col">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Covariates:</label>
                    <div class="aw-checkbox-list" style="max-height:150px;overflow-y:auto;">
                        ${predChecks || '<span style="color:#7a8f7a;">No columns available</span>'}
                    </div>
                    <div style="margin-top:6px;">
                        <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('covariates', true)">All</button>
                        <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('covariates', false)">Clear</button>
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Confidence level:</label>
                    <select id="cfg-conf">
                        <option value="95">95%</option>
                        <option value="99">99%</option>
                        <option value="90">90%</option>
                    </select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Cox PH:</strong> Semi-parametric survival regression. Estimates hazard ratios (HR &gt; 1 = higher risk). Categorical covariates are automatically dummy-coded.
                </div>
            `;
        }
        if (id === 'discriminant_analysis') {
            const predChecks = columns.filter(c => c.dtype === 'numeric').map(c => `
                <label class="aw-checkbox-item"><input type="checkbox" name="predictors" value="${c.name}"><span>${c.name}</span></label>
            `).join('');
            const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
            return `
                <div class="aw-form-group">
                    <label>Group variable (target):</label>
                    <select id="cfg-response">${catCols || colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Predictors:</label>
                    <div class="aw-checkbox-list" style="max-height:150px;overflow-y:auto;">
                        ${predChecks || '<span style="color:#7a8f7a;">No numeric columns</span>'}
                    </div>
                    <div style="margin-top:6px;">
                        <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('predictors', true)">All</button>
                        <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('predictors', false)">Clear</button>
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Method:</label>
                    <select id="cfg-method">
                        <option value="lda">LDA (Linear Discriminant Analysis)</option>
                        <option value="qda">QDA (Quadratic Discriminant Analysis)</option>
                    </select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Discriminant Analysis:</strong> LDA assumes equal covariances (linear boundary), QDA allows class-specific covariances (curved boundary). Use LDA when groups have similar spread.
                </div>
            `;
        }
        if (id === 'acceptance_sampling') {
            return `
                <div class="aw-form-group">
                    <label>Plan type:</label>
                    <select id="cfg-plan_type">
                        <option value="single">Single Sampling</option>
                        <option value="double">Double Sampling</option>
                    </select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Sample size (n):</label>
                        <input type="number" id="cfg-sample_size" value="50" min="1" max="10000">
                    </div>
                    <div class="aw-form-group">
                        <label>Accept number (Ac):</label>
                        <input type="number" id="cfg-accept_number" value="2" min="0" max="100">
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Lot size (N):</label>
                        <input type="number" id="cfg-lot_size" value="1000" min="1" max="1000000">
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>AQL (%):</label>
                        <input type="number" id="cfg-aql" value="1" min="0.01" max="10" step="0.1">
                    </div>
                    <div class="aw-form-group">
                        <label>LTPD (%):</label>
                        <input type="number" id="cfg-ltpd" value="5" min="0.1" max="50" step="0.1">
                    </div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Acceptance Sampling:</strong>
                    AQL = max acceptable defect rate (producer's quality target).
                    LTPD = worst tolerable defect rate (consumer's protection).
                    OC curve shows P(accept) at each defect rate. No dataset required.
                </div>
            `;
        }
        if (id === 'prop_1sample') {
            return `
                <div class="aw-form-group">
                    <label>Variable (binary / categorical):</label>
                    <select id="cfg-var">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Event value (count as success):</label>
                    <input type="text" id="cfg-event" placeholder="e.g. 1, Yes, Pass (blank = 1 for binary)">
                </div>
                <div class="aw-form-group">
                    <label>Hypothesized proportion (p₀):</label>
                    <input type="number" id="cfg-p0" value="0.5" min="0" max="1" step="0.01">
                </div>
                <div class="aw-form-group">
                    <label>Alternative:</label>
                    <select id="cfg-alternative">
                        <option value="two-sided">Two-sided (p ≠ p₀)</option>
                        <option value="greater">Greater (p > p₀)</option>
                        <option value="less">Less (p < p₀)</option>
                    </select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">1-Proportion Z-Test:</strong>
                    Tests whether an observed proportion differs from a target. Uses normal approximation with Wilson confidence interval.
                </div>
            `;
        }
        if (id === 'prop_2sample') {
            return `
                <div class="aw-form-group">
                    <label>Variable (binary / categorical):</label>
                    <select id="cfg-var">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Group variable:</label>
                    <select id="cfg-group_var">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Event value (count as success):</label>
                    <input type="text" id="cfg-event" placeholder="e.g. 1, Yes, Pass (blank = 1 for binary)">
                </div>
                <div class="aw-form-group">
                    <label>Alternative:</label>
                    <select id="cfg-alternative">
                        <option value="two-sided">Two-sided (p₁ ≠ p₂)</option>
                        <option value="greater">Greater (p₁ > p₂)</option>
                        <option value="less">Less (p₁ < p₂)</option>
                    </select>
                </div>
                <div style="background:rgba(74,144,217,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#4a90d9;">2-Proportion Z-Test:</strong>
                    Compares proportions between two groups using pooled Z-statistic. Group variable must have exactly 2 levels.
                </div>
            `;
        }
        if (id === 'fisher_exact') {
            return `
                <div class="aw-form-group">
                    <label>Row variable:</label>
                    <select id="cfg-var">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Column variable:</label>
                    <select id="cfg-var2">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Alternative:</label>
                    <select id="cfg-alternative">
                        <option value="two-sided">Two-sided</option>
                        <option value="greater">Greater</option>
                        <option value="less">Less</option>
                    </select>
                </div>
                <div style="background:rgba(232,149,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e89547;">Fisher's Exact Test:</strong>
                    Exact test for 2×2 contingency tables. Preferred over chi-square when expected cell counts &lt; 5. Reports odds ratio with CI.
                </div>
            `;
        }
        if (id === 'poisson_1sample') {
            return `
                <div class="aw-form-group">
                    <label>Count variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Hypothesized rate (λ₀):</label>
                        <input type="number" id="cfg-rate0" value="1.0" min="0" step="0.1">
                    </div>
                    <div class="aw-form-group">
                        <label>Exposure (time/units):</label>
                        <input type="number" id="cfg-exposure" value="1.0" min="0.001" step="0.1">
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Alternative:</label>
                    <select id="cfg-alternative">
                        <option value="two-sided">Two-sided (λ ≠ λ₀)</option>
                        <option value="greater">Greater (λ > λ₀)</option>
                        <option value="less">Less (λ < λ₀)</option>
                    </select>
                </div>
                <div style="background:rgba(217,74,74,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#d94a4a;">Poisson Rate Test:</strong>
                    Tests if an observed event count (summed over the variable) differs from a hypothesized rate × exposure. Uses exact Poisson test.
                </div>
            `;
        }
        if (id === 'power_z') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>Difference (δ):</label><input type="number" id="cfg-delta" value="0.5" step="0.1"></div>
                    <div class="aw-form-group"><label>Std Dev (σ):</label><input type="number" id="cfg-sigma" value="1.0" step="0.1"></div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>Alpha:</label><input type="number" id="cfg-alpha" value="0.05" step="0.01" min="0.001" max="0.5"></div>
                    <div class="aw-form-group"><label>Power:</label><input type="number" id="cfg-power" value="0.80" step="0.05" min="0.1" max="0.99"></div>
                </div>
                <div style="background:rgba(74,144,217,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#4a90d9;">1-Sample Z Power:</strong> Finds sample size to detect δ with given power. No dataset needed.
                </div>
            `;
        }
        if (id === 'power_1prop') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>Null proportion (p₀):</label><input type="number" id="cfg-p0" value="0.5" step="0.01" min="0" max="1"></div>
                    <div class="aw-form-group"><label>Alt proportion (pₐ):</label><input type="number" id="cfg-pa" value="0.65" step="0.01" min="0" max="1"></div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>Alpha:</label><input type="number" id="cfg-alpha" value="0.05" step="0.01"></div>
                    <div class="aw-form-group"><label>Power:</label><input type="number" id="cfg-power" value="0.80" step="0.05"></div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">1-Proportion Power:</strong> Sample size to detect p=pₐ vs p₀. No dataset needed.
                </div>
            `;
        }
        if (id === 'power_2prop') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>Proportion 1 (p₁):</label><input type="number" id="cfg-p1" value="0.5" step="0.01"></div>
                    <div class="aw-form-group"><label>Proportion 2 (p₂):</label><input type="number" id="cfg-p2" value="0.65" step="0.01"></div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>Alpha:</label><input type="number" id="cfg-alpha" value="0.05" step="0.01"></div>
                    <div class="aw-form-group"><label>Power:</label><input type="number" id="cfg-power" value="0.80" step="0.05"></div>
                </div>
                <div class="aw-form-group"><label>n₂/n₁ ratio:</label><input type="number" id="cfg-ratio" value="1.0" step="0.1" min="0.1"></div>
                <div style="background:rgba(74,144,217,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#4a90d9;">2-Proportion Power:</strong> Sample size per group to detect difference. No dataset needed.
                </div>
            `;
        }
        if (id === 'power_1variance') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>Null σ₀:</label><input type="number" id="cfg-sigma0" value="1.0" step="0.1"></div>
                    <div class="aw-form-group"><label>Alt σ₁:</label><input type="number" id="cfg-sigma1" value="1.5" step="0.1"></div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>Alpha:</label><input type="number" id="cfg-alpha" value="0.05" step="0.01"></div>
                    <div class="aw-form-group"><label>Power:</label><input type="number" id="cfg-power" value="0.80" step="0.05"></div>
                </div>
                <div style="background:rgba(232,149,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e89547;">1-Variance Power:</strong> Sample size to detect σ₁ vs σ₀ using chi-square test. No dataset needed.
                </div>
            `;
        }
        if (id === 'power_2variance') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>Variance ratio (σ₁²/σ₂²):</label><input type="number" id="cfg-variance_ratio" value="2.0" step="0.1" min="0.1"></div>
                    <div class="aw-form-group"><label>n₂/n₁ ratio:</label><input type="number" id="cfg-ratio" value="1.0" step="0.1" min="0.1"></div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>Alpha:</label><input type="number" id="cfg-alpha" value="0.05" step="0.01"></div>
                    <div class="aw-form-group"><label>Power:</label><input type="number" id="cfg-power" value="0.80" step="0.05"></div>
                </div>
                <div style="background:rgba(232,149,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e89547;">2-Variance Power:</strong> Sample size per group to detect variance ratio using F-test. No dataset needed.
                </div>
            `;
        }
        if (id === 'power_equivalence') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>True difference (δ):</label><input type="number" id="cfg-delta" value="0.0" step="0.1"></div>
                    <div class="aw-form-group"><label>Equiv margin (±):</label><input type="number" id="cfg-margin" value="0.5" step="0.1"></div>
                </div>
                <div class="aw-form-group"><label>Std Dev (σ):</label><input type="number" id="cfg-sigma" value="1.0" step="0.1"></div>
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>Alpha:</label><input type="number" id="cfg-alpha" value="0.05" step="0.01"></div>
                    <div class="aw-form-group"><label>Power:</label><input type="number" id="cfg-power" value="0.80" step="0.05"></div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Equivalence Power:</strong> Sample size for TOST to prove equivalence within ±margin. No dataset needed.
                </div>
            `;
        }
        if (id === 'power_doe') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>Number of factors (k):</label><input type="number" id="cfg-factors" value="3" step="1" min="2" max="8"></div>
                    <div class="aw-form-group"><label>Min effect (Δ):</label><input type="number" id="cfg-delta" value="1.0" step="0.1"></div>
                </div>
                <div class="aw-form-group"><label>Std Dev (σ):</label><input type="number" id="cfg-sigma" value="1.0" step="0.1"></div>
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>Alpha:</label><input type="number" id="cfg-alpha" value="0.05" step="0.01"></div>
                    <div class="aw-form-group"><label>Power:</label><input type="number" id="cfg-power" value="0.80" step="0.05"></div>
                </div>
                <div style="background:rgba(74,144,217,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#4a90d9;">DOE Factorial Power:</strong> Required replicates for 2^k full factorial design. No dataset needed.
                </div>
            `;
        }
        if (id === 'sample_size_ci') {
            return `
                <div class="aw-form-group">
                    <label>Estimate type:</label>
                    <select id="cfg-type">
                        <option value="mean">Mean</option>
                        <option value="proportion">Proportion</option>
                    </select>
                </div>
                <div class="aw-form-group"><label>Target half-width (margin of error):</label><input type="number" id="cfg-half_width" value="0.5" step="0.1" min="0.001"></div>
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>σ (for mean):</label><input type="number" id="cfg-sigma" value="1.0" step="0.1"></div>
                    <div class="aw-form-group"><label>p est (for proportion):</label><input type="number" id="cfg-p_est" value="0.5" step="0.01" min="0" max="1"></div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Sample Size for CI:</strong> How many observations to achieve a target confidence interval width. No dataset needed.
                </div>
            `;
        }
        if (id === 'sample_size_tolerance') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group"><label>Coverage (e.g. 0.99):</label><input type="number" id="cfg-coverage" value="0.99" step="0.01" min="0.5" max="0.999"></div>
                    <div class="aw-form-group"><label>Confidence (e.g. 0.95):</label><input type="number" id="cfg-confidence" value="0.95" step="0.01" min="0.5" max="0.999"></div>
                </div>
                <div class="aw-form-group">
                    <label>Interval type:</label>
                    <select id="cfg-type">
                        <option value="two-sided">Two-sided</option>
                        <option value="one-sided">One-sided</option>
                    </select>
                </div>
                <div style="background:rgba(217,74,74,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#d94a4a;">Tolerance Interval Size:</strong> Sample size so the tolerance k-factor converges. E.g. 99%/95% = 99% of population covered with 95% confidence. No dataset needed.
                </div>
            `;
        }
        if (id === 'sem') {
            return `
                <div class="aw-form-group">
                    <label>Model type:</label>
                    <select id="cfg-model_type">
                        <option value="path">Path Analysis (direct effects)</option>
                        <option value="mediation">Mediation (X → M → Y)</option>
                    </select>
                </div>
                <div class="aw-form-group">
                    <label>Outcome (Y):</label>
                    <select id="cfg-outcome">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Predictors (X):
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('predictors', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('predictors', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:100px;overflow-y:auto;">
                        ${numericCols.map(c => `<label><input type="checkbox" name="predictors" value="${c.name}"> ${c.name}</label>`).join('')}
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Mediator (M) - for mediation model:</label>
                    <select id="cfg-mediator"><option value="">None</option>${numCols}</select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Causal Paths:</strong> SEM tests hypothesized causal relationships. Mediation analysis decomposes effects into direct and indirect paths.
                </div>
            `;
        }
        if (id === 'regression_ml') {
            return `
                <div class="aw-form-group">
                    <label>Target (response):</label>
                    <select id="cfg-target">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Features:
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">${featureCheckboxes}</div>
                </div>
                <div class="aw-form-group">
                    <label>Algorithm:</label>
                    <select id="cfg-algorithm">
                        <option value="random_forest">Random Forest</option>
                        <option value="gradient_boosting">Gradient Boosting</option>
                        <option value="svr">SVR</option>
                        <option value="linear">Linear Regression</option>
                    </select>
                </div>
                <div class="aw-form-group">
                    <label>Test split:</label>
                    <select id="cfg-split">
                        <option value="0.2">20%</option>
                        <option value="0.3">30%</option>
                        <option value="0.25">25%</option>
                    </select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">ML Regression:</strong> Predicts continuous target. Reports RMSE, R², and feature importance.
                </div>
            `;
        }
        if (id === 'xgboost' || id === 'lightgbm') {
            const boostName = id === 'xgboost' ? 'XGBoost' : 'LightGBM';
            const boostNote = id === 'xgboost'
                ? 'Gradient-boosted trees. Strong baseline for tabular data. Tune n_estimators and max_depth first.'
                : 'Fast gradient boosting with leaf-wise growth. Often faster than XGBoost on large datasets. Tune num_leaves first.';
            return `
                <div class="aw-form-group">
                    <label>Target:</label>
                    <select id="cfg-target">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Features:
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">${featureCheckboxes}</div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Estimators:</label>
                        <input type="number" id="cfg-n_estimators" value="100" min="10" max="2000">
                    </div>
                    <div class="aw-form-group">
                        <label>${id === 'lightgbm' ? 'Num leaves:' : 'Max depth:'}</label>
                        <input type="number" id="cfg-${id === 'lightgbm' ? 'num_leaves' : 'max_depth'}" value="${id === 'lightgbm' ? '31' : '6'}" min="2" max="${id === 'lightgbm' ? '256' : '20'}">
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Learning rate:</label>
                    <input type="number" id="cfg-learning_rate" value="0.1" step="0.01" min="0.001" max="1">
                </div>
                <div class="aw-form-group">
                    <label>Task:</label>
                    <select id="cfg-task_type">
                        <option value="auto">Auto-detect</option>
                        <option value="classification">Classification</option>
                        <option value="regression">Regression</option>
                    </select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">${boostName}:</strong> ${boostNote}
                </div>
            `;
        }
        if (id === 'model_compare') {
            return `
                <div class="aw-form-group">
                    <label>Target:</label>
                    <select id="cfg-target">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Features:
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">${featureCheckboxes}</div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>CV folds:</label>
                        <input type="number" id="cfg-cv_folds" value="5" min="2" max="20">
                    </div>
                    <div class="aw-form-group">
                        <label>Task:</label>
                        <select id="cfg-task_type">
                            <option value="auto">Auto-detect</option>
                            <option value="classification">Classification</option>
                            <option value="regression">Regression</option>
                        </select>
                    </div>
                </div>
                <div style="background:rgba(74,144,217,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#4a90d9;">Model Compare:</strong> Benchmarks RF, XGBoost, LightGBM, SVM, and Linear/Logistic with cross-validation. Shows ranked performance.
                </div>
            `;
        }
        if (id === 'shap_explain') {
            return `
                <div class="aw-form-group">
                    <label>Target:</label>
                    <select id="cfg-target">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Features:
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">${featureCheckboxes}</div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">SHAP Explain:</strong> Builds a model internally and computes SHAP values showing how each feature drives predictions. Beeswarm + bar plots.
                </div>
            `;
        }
        if (id === 'hyperparameter_tune') {
            return `
                <div class="aw-form-group">
                    <label>Target:</label>
                    <select id="cfg-target">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Features:
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">${featureCheckboxes}</div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Model:</label>
                        <select id="cfg-model_type">
                            <option value="random_forest">Random Forest</option>
                            <option value="xgboost">XGBoost</option>
                            <option value="lightgbm">LightGBM</option>
                            <option value="svm">SVM</option>
                        </select>
                    </div>
                    <div class="aw-form-group">
                        <label>Trials:</label>
                        <input type="number" id="cfg-n_trials" value="50" min="10" max="500">
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Task:</label>
                    <select id="cfg-task_type">
                        <option value="auto">Auto-detect</option>
                        <option value="classification">Classification</option>
                        <option value="regression">Regression</option>
                    </select>
                </div>
                <div style="background:rgba(232,149,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e89547;">Hyperparameter Tune:</strong> Bayesian optimization (Optuna) to find best hyperparameters. More efficient than grid search.
                </div>
            `;
        }
        if (id === 'factor_analysis') {
            return `
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Variables:
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">${numericFeatureCheckboxes}</div>
                </div>
                <div class="aw-form-group">
                    <label>Number of factors:</label>
                    <input type="number" id="cfg-n_components" value="3" min="1" max="20">
                </div>
                <div style="background:rgba(74,144,217,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#4a90d9;">Factor Analysis:</strong> Identifies latent constructs from observed variables. Use KMO ≥ 0.6 and Bartlett p &lt; 0.05 to validate suitability.
                </div>
            `;
        }
        if (id === 'correspondence_analysis') {
            return `
                <div class="aw-form-group">
                    <label>Row variable:</label>
                    <select id="cfg-row_var">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Column variable:</label>
                    <select id="cfg-col_var">${colOptions}</select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Correspondence Analysis:</strong> Visualizes associations between categorical variables in 2D. Points close together are associated.
                </div>
            `;
        }
        if (id === 'item_analysis') {
            return `
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Items (scale variables):
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:150px;overflow-y:auto;">${numericFeatureCheckboxes}</div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Item Analysis:</strong> Cronbach's α for internal consistency. Item-total correlations show which items to drop. α ≥ 0.7 is acceptable.
                </div>
            `;
        }
    }

    // Bayesian Inference section - feeds Synara
    if (type === 'bayesian') {
        const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
        const numericCols = columns.filter(c => c.dtype === 'numeric');
        const numericFeatureCheckboxes = numericCols.map(c => `
            <label class="aw-checkbox-item">
                <input type="checkbox" name="features" value="${c.name}" checked>
                <span>${c.name}</span>
            </label>
        `).join('');

        const synaraNote = `
            <div style="background:rgba(74,159,110,0.15);padding:10px;border-radius:4px;font-size:10px;color:var(--aw-text);margin-top:8px;border-left:3px solid var(--aw-accent);">
                <strong style="color:var(--aw-accent);">Synara Integration:</strong> Results feed into hypothesis testing. Select a project above to link evidence.
            </div>
        `;

        if (id === 'bayes_regression') {
            return `
                <div class="aw-form-group">
                    <label>Target (response):</label>
                    <select id="cfg-target">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;justify-content:space-between;align-items:center;">
                        Predictors:
                        <span style="font-size:9px;font-weight:normal;">
                            <a href="#" onclick="toggleAllCheckboxes('features', true);return false;" style="color:var(--aw-accent);">All</a> |
                            <a href="#" onclick="toggleAllCheckboxes('features', false);return false;" style="color:var(--aw-accent);">None</a>
                        </span>
                    </label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">
                        ${numericFeatureCheckboxes}
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Credible interval:</label>
                    <select id="cfg-ci">
                        <option value="0.95">95%</option>
                        <option value="0.90">90%</option>
                        <option value="0.99">99%</option>
                    </select>
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_ttest') {
            return `
                ${twoSampleLayout(numCols, colOptions, {label1: 'Sample 1', label2: 'Sample 2', uid: 'bt'})}
                <div class="aw-form-group">
                    <label>Prior effect size (Cohen's d):</label>
                    <select id="cfg-prior_scale">
                        <option value="medium" selected>Medium (0.5)</option>
                        <option value="small">Small (0.2)</option>
                        <option value="large">Large (0.8)</option>
                        <option value="ultrawide">Ultrawide (1.0)</option>
                    </select>
                </div>
                <div class="aw-form-group">
                    <label>Credible interval:</label>
                    <select id="cfg-ci">
                        <option value="0.95">95%</option>
                        <option value="0.90">90%</option>
                    </select>
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_ab') {
            return `
                <div class="aw-form-group">
                    <label>Group column:</label>
                    <select id="cfg-group">${catCols || colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Success column (binary 0/1):</label>
                    <select id="cfg-success">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Prior (Beta distribution):</label>
                    <select id="cfg-prior">
                        <option value="uniform">Uniform (1, 1)</option>
                        <option value="jeffreys">Jeffreys (0.5, 0.5)</option>
                        <option value="informed">Informed (5, 5)</option>
                    </select>
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_correlation') {
            return `
                <div class="aw-form-group">
                    <label>Variable X:</label>
                    <select id="cfg-var1">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Variable Y:</label>
                    <select id="cfg-var2">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Prior:</label>
                    <select id="cfg-prior">
                        <option value="uniform">Uniform [-1, 1]</option>
                        <option value="stretched_beta">Stretched Beta (moderate)</option>
                    </select>
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_anova') {
            return `
                <div class="aw-form-group">
                    <label>Response (measurement):</label>
                    <select id="cfg-response">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Factor (groups):</label>
                    <select id="cfg-factor">${catCols || colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Prior scale:</label>
                    <select id="cfg-prior_scale">
                        <option value="medium">Medium</option>
                        <option value="wide">Wide</option>
                        <option value="ultrawide">Ultrawide</option>
                    </select>
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_changepoint') {
            return `
                <div class="aw-form-group">
                    <label>Variable to analyze:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Time/sequence column (optional):</label>
                    <select id="cfg-time"><option value="">Row index</option>${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Max change points:</label>
                    <select id="cfg-max_cp">
                        <option value="1">1</option>
                        <option value="2" selected>2</option>
                        <option value="3">3</option>
                    </select>
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_proportion') {
            return `
                <div class="aw-form-group">
                    <label>Success column (binary 0/1):</label>
                    <select id="cfg-success">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Prior belief (Beta distribution):</label>
                    <select id="cfg-prior">
                        <option value="uniform">No prior info (1, 1)</option>
                        <option value="jeffreys">Jeffreys (0.5, 0.5)</option>
                        <option value="optimistic">Optimistic (8, 2)</option>
                        <option value="pessimistic">Pessimistic (2, 8)</option>
                    </select>
                </div>
                <div class="aw-form-group">
                    <label>Credible interval:</label>
                    <select id="cfg-ci">
                        <option value="0.95">95%</option>
                        <option value="0.90">90%</option>
                    </select>
                </div>
                ${synaraNote}
            `;
        }
        // Bayesian SPC — measurement + spec limits
        if (id === 'bayes_spc_capability' || id === 'bayes_cpk_predict') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>LSL:</label>
                        <input type="number" id="cfg-lsl" step="any" placeholder="Lower spec">
                    </div>
                    <div class="aw-form-group">
                        <label>USL:</label>
                        <input type="number" id="cfg-usl" step="any" placeholder="Upper spec">
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Target (optional):</label>
                    <input type="number" id="cfg-target" step="any" placeholder="Nominal">
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_spc_changepoint') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Hazard (prior change rate):</label>
                    <select id="cfg-hazard">
                        <option value="0.01">Low (1/100)</option>
                        <option value="0.05" selected>Medium (1/20)</option>
                        <option value="0.1">High (1/10)</option>
                    </select>
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_spc_control') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Subgroup column (optional):</label>
                    <select id="cfg-subgroup"><option value="">None (individuals)</option>${colOptions}</select>
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_spc_acceptance') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>LSL:</label>
                        <input type="number" id="cfg-lsl" step="any" placeholder="Lower spec">
                    </div>
                    <div class="aw-form-group">
                        <label>USL:</label>
                        <input type="number" id="cfg-usl" step="any" placeholder="Upper spec">
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Lot size:</label>
                    <input type="number" id="cfg-lot_size" value="1000">
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_msa') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Part column:</label>
                    <select id="cfg-part">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Operator column:</label>
                    <select id="cfg-operator">${colOptions}</select>
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_chi2') {
            return `
                <div class="aw-form-group">
                    <label>Row variable:</label>
                    <select id="cfg-row_var">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Column variable:</label>
                    <select id="cfg-col_var">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Prior scale:</label>
                    <select id="cfg-prior_scale">
                        <option value="medium">Medium</option>
                        <option value="wide">Wide</option>
                        <option value="ultrawide">Ultrawide</option>
                    </select>
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_poisson') {
            return `
                <div class="aw-form-group">
                    <label>Count variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Prior α:</label>
                        <input type="number" id="cfg-prior_a" value="1" step="0.1" min="0.01">
                    </div>
                    <div class="aw-form-group">
                        <label>Prior β:</label>
                        <input type="number" id="cfg-prior_b" value="1" step="0.1" min="0.01">
                    </div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Bayesian Poisson:</strong> Gamma-Poisson conjugate model. Prior α=β=1 is uninformative. Posterior gives credible interval for rate λ.
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_logistic') {
            return `
                <div class="aw-form-group">
                    <label>Outcome (binary):</label>
                    <select id="cfg-var1">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Predictor:</label>
                    <select id="cfg-var2">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Exposure column (optional):</label>
                    <select id="cfg-exposure"><option value="">None</option>${numCols}</select>
                </div>
                <div style="background:rgba(74,144,217,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#4a90d9;">Bayesian Logistic:</strong> Binary outcome regression with posterior odds ratios and credible intervals. More interpretable than p-values for rare events.
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_meta') {
            return `
                <div class="aw-form-group">
                    <label>Effect sizes column:</label>
                    <select id="cfg-effects_col">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Standard errors column:</label>
                    <select id="cfg-se_col">${numCols}</select>
                </div>
                <div style="background:rgba(232,149,71,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:#e89547;">Bayesian Meta-Analysis:</strong> Hierarchical model pooling multiple studies. Posterior gives credible interval for overall effect + heterogeneity τ².
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_ewma') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Target (0 = mean):</label>
                        <input type="number" id="cfg-target" value="0" step="any">
                    </div>
                    <div class="aw-form-group">
                        <label>λ (smoothing):</label>
                        <input type="number" id="cfg-lambda_param" value="0.2" step="0.05" min="0.05" max="1">
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Prior scale:</label>
                    <select id="cfg-prior_scale">
                        <option value="medium">Medium</option>
                        <option value="wide">Wide</option>
                    </select>
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_equivalence') {
            return `
                <div class="aw-form-group">
                    <label>Variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Group variable:</label>
                    <select id="cfg-group">${colOptions}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>ROPE low:</label>
                        <input type="number" id="cfg-rope_low" value="-0.1" step="0.01">
                    </div>
                    <div class="aw-form-group">
                        <label>ROPE high:</label>
                        <input type="number" id="cfg-rope_high" value="0.1" step="0.01">
                    </div>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Bayesian Equivalence:</strong> Tests if difference falls within ROPE (Region of Practical Equivalence). More useful than null hypothesis testing for process validation.
                </div>
                ${synaraNote}
            `;
        }
        if (id === 'bayes_survival') {
            return `
                <div class="aw-form-group">
                    <label>Time variable:</label>
                    <select id="cfg-var1">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Event indicator (1=event):</label>
                    <select id="cfg-var2"><option value="">All observed</option>${numCols}</select>
                </div>
                <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                    <strong style="color:var(--aw-accent);">Bayesian Survival:</strong> Posterior distribution for survival parameters with credible intervals. Handles censoring naturally.
                </div>
                ${synaraNote}
            `;
        }
        // Bayesian reliability forms — measurement or time-to-event
        if (id.startsWith('bayes_reliability') || id === 'bayes_rul' || id === 'bayes_alt' || id === 'bayes_repairable' || id === 'bayes_warranty' || id === 'bayes_competing_risks' || id === 'bayes_spares' || id === 'bayes_system_reliability') {
            return `
                <div class="aw-form-group">
                    <label>Time / measurement column:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Event column (1=event, 0=censored):</label>
                    <select id="cfg-event"><option value="">All observed</option>${numCols}</select>
                </div>
                ${synaraNote}
            `;
        }
        return `<p style="color:#9aaa9a;font-size:11px;">Select a Bayesian analysis.</p>`;
    }

    if (type === 'viz') {
        if (id === 'histogram' || id === 'boxplot') {
            return `
                <div class="aw-form-group">
                    <label>Variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Group by (color):</label>
                    <select id="cfg-by"><option value="">None</option>${colOptions}</select>
                </div>
            `;
        }
        if (id === 'scatter') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>X variable:</label>
                        <select id="cfg-x">${numCols}</select>
                    </div>
                    <div class="aw-form-group">
                        <label>Y variable:</label>
                        <select id="cfg-y">${numCols}</select>
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Color by:</label>
                    <select id="cfg-color"><option value="">None</option>${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;align-items:center;gap:8px;text-transform:none;">
                        <input type="checkbox" id="cfg-trendline" style="width:auto;">
                        Show trendline
                    </label>
                </div>
            `;
        }
        if (id === 'matrix') {
            return `
                <div class="aw-form-group">
                    <label>Variables:</label>
                    <select id="cfg-vars" multiple style="height:120px;">${numCols}</select>
                    <small style="color:#7a8f7a;">Select 2-6 variables for matrix</small>
                </div>
                <div class="aw-form-group">
                    <label>Color by:</label>
                    <select id="cfg-color"><option value="">None</option>${colOptions}</select>
                </div>
            `;
        }
        if (id === 'timeseries') {
            return `
                <div class="aw-form-group">
                    <label>Time/Index column:</label>
                    <select id="cfg-x">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Value column(s):</label>
                    <select id="cfg-y" multiple style="height:100px;">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;align-items:center;gap:8px;text-transform:none;">
                        <input type="checkbox" id="cfg-markers" style="width:auto;">
                        Show markers
                    </label>
                </div>
            `;
        }
        if (id === 'probability') {
            return `
                <div class="aw-form-group">
                    <label>Variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Distribution:</label>
                    <select id="cfg-dist">
                        <option value="norm">Normal</option>
                        <option value="lognorm">Lognormal</option>
                        <option value="expon">Exponential</option>
                        <option value="weibull">Weibull</option>
                    </select>
                </div>
            `;
        }
        if (id === 'pareto') {
            const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
            return `
                <div class="aw-form-group">
                    <label>Category column:</label>
                    <select id="cfg-category">${catCols || colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Count/Value column (optional):</label>
                    <select id="cfg-value"><option value="">Count occurrences</option>${numCols}</select>
                </div>
            `;
        }
        if (id === 'heatmap') {
            return `
                <div class="aw-form-group">
                    <label>Variables:</label>
                    <select id="cfg-vars" multiple style="height:120px;">${numCols}</select>
                    <small style="color:#7a8f7a;">Select numeric columns for correlation heatmap</small>
                </div>
            `;
        }
    }

    // Data Tools
    if (type === 'tools') {
        if (id === 'calculator') {
            return `
                <div class="aw-form-group">
                    <label>New column name:</label>
                    <input type="text" id="cfg-new_col" placeholder="e.g., Total">
                </div>
                <div class="aw-form-group">
                    <label>Expression:</label>
                    <input type="text" id="cfg-expression" placeholder="e.g., Price * Quantity">
                    <small style="color:#7a8f7a;">Use column names, operators (+, -, *, /), and functions (log, sqrt, abs)</small>
                </div>
                <div class="aw-form-group">
                    <label>Available columns:</label>
                    <div style="max-height:100px;overflow-y:auto;background:#0a0f0a;padding:6px;border-radius:3px;font-size:10px;color:#9aaa9a;">
                        ${columns.map(c => c.name).join(', ')}
                    </div>
                </div>
            `;
        }
        if (id === 'subset') {
            return `
                <div class="aw-form-group">
                    <label>Filter column:</label>
                    <select id="cfg-filter_col">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Condition:</label>
                    <select id="cfg-condition">
                        <option value="eq">Equals (==)</option>
                        <option value="ne">Not equals (!=)</option>
                        <option value="gt">Greater than (>)</option>
                        <option value="gte">Greater or equal (>=)</option>
                        <option value="lt">Less than (<)</option>
                        <option value="lte">Less or equal (<=)</option>
                        <option value="contains">Contains</option>
                        <option value="notna">Not empty</option>
                        <option value="isna">Is empty</option>
                    </select>
                </div>
                <div class="aw-form-group" id="cfg-value-group">
                    <label>Value:</label>
                    <input type="text" id="cfg-filter_value" placeholder="Filter value">
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;align-items:center;gap:8px;text-transform:none;">
                        <input type="checkbox" id="cfg-new_tab" checked style="width:auto;">
                        Create as new tab
                    </label>
                </div>
            `;
        }
        if (id === 'sort') {
            return `
                <div class="aw-form-group">
                    <label>Sort by:</label>
                    <select id="cfg-sort_col">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Order:</label>
                    <select id="cfg-order">
                        <option value="asc">Ascending (A→Z, 1→9)</option>
                        <option value="desc">Descending (Z→A, 9→1)</option>
                    </select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;align-items:center;gap:8px;text-transform:none;">
                        <input type="checkbox" id="cfg-new_tab" style="width:auto;">
                        Create as new tab
                    </label>
                </div>
            `;
        }
        if (id === 'transpose') {
            return `
                <div class="aw-form-group">
                    <p style="color:#9aaa9a;margin:0 0 12px 0;">Transpose rows and columns. The first column will become the header.</p>
                    <label style="display:flex;align-items:center;gap:8px;text-transform:none;">
                        <input type="checkbox" id="cfg-new_tab" checked style="width:auto;">
                        Create as new tab
                    </label>
                </div>
            `;
        }
        if (id === 'stack') {
            return `
                <div class="aw-form-group">
                    <label>Operation:</label>
                    <select id="cfg-operation">
                        <option value="melt">Unpivot (wide → long)</option>
                        <option value="pivot">Pivot (long → wide)</option>
                    </select>
                </div>
                <div class="aw-form-group" id="cfg-melt-options">
                    <label>ID columns (keep as-is):</label>
                    <select id="cfg-id_cols" multiple style="height:80px;">${colOptions}</select>
                    <small style="color:#7a8f7a;">Columns that identify each row</small>
                </div>
                <div class="aw-form-group" id="cfg-pivot-options" style="display:none;">
                    <label>Index column:</label>
                    <select id="cfg-index">${colOptions}</select>
                    <label style="margin-top:8px;">Columns from:</label>
                    <select id="cfg-pivot_col">${colOptions}</select>
                    <label style="margin-top:8px;">Values from:</label>
                    <select id="cfg-values">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label style="display:flex;align-items:center;gap:8px;text-transform:none;">
                        <input type="checkbox" id="cfg-new_tab" checked style="width:auto;">
                        Create as new tab
                    </label>
                </div>
            `;
        }
    }

    // --- ANOVA extensions ---
    if (id === 'split_plot_anova') {
        const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
        return `
            <div class="aw-form-group">
                <label>Response:</label>
                <select id="cfg-response">${numCols}</select>
            </div>
            <div class="aw-form-group">
                <label>Whole-plot factor (hard-to-change):</label>
                <select id="cfg-whole_plot_factors">${catCols || colOptions}</select>
            </div>
            <div class="aw-form-group">
                <label>Sub-plot factor (easy-to-change):</label>
                <select id="cfg-sub_plot_factors">${catCols || colOptions}</select>
            </div>
            <div class="aw-form-group">
                <label>Block / whole-plot ID:</label>
                <select id="cfg-block">${catCols || colOptions}</select>
            </div>
        `;
    }
    if (id === 'repeated_measures_anova') {
        const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
        return `
            <div class="aw-form-group">
                <label>Response:</label>
                <select id="cfg-response">${numCols}</select>
            </div>
            <div class="aw-form-group">
                <label>Subject ID:</label>
                <select id="cfg-subject">${catCols || colOptions}</select>
            </div>
            <div class="aw-form-group">
                <label>Within-subject factor:</label>
                <select id="cfg-within_factor">${catCols || colOptions}</select>
            </div>
            <div class="aw-form-group">
                <label>Between-subject factor (optional):</label>
                <select id="cfg-between_factor"><option value="">(none)</option>${(catCols || colOptions)}</select>
            </div>
        `;
    }
    if (id === 'anom') {
        const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
        return `
            <div class="aw-form-group">
                <label>Response:</label>
                <select id="cfg-var">${numCols}</select>
            </div>
            <div class="aw-form-group">
                <label>Factor (groups):</label>
                <select id="cfg-factor">${catCols || colOptions}</select>
            </div>
            <div class="aw-form-group">
                <label>Confidence level:</label>
                <select id="cfg-conf">
                    <option value="95">95%</option>
                    <option value="99">99%</option>
                    <option value="90">90%</option>
                </select>
            </div>
        `;
    }
    if (id === 'glm') {
        const catCols = columns.filter(c => c.dtype === 'text');
        const catCheckboxes = catCols.map(c => `
            <label class="aw-checkbox-item"><input type="checkbox" name="fixed_factors" value="${c.name}"><span>${c.name}</span></label>
        `).join('') || '<span style="color:#7a8f7a;">No categorical columns</span>';
        const numericCols = columns.filter(c => c.dtype === 'numeric');
        const covCheckboxes = numericCols.map(c => `
            <label class="aw-checkbox-item"><input type="checkbox" name="covariates" value="${c.name}"><span>${c.name}</span></label>
        `).join('');
        return `
            <div class="aw-form-group">
                <label>Response:</label>
                <select id="cfg-response">${numCols}</select>
            </div>
            <div class="aw-form-group">
                <label>Fixed factors:</label>
                <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">${catCheckboxes}</div>
            </div>
            <div class="aw-form-group">
                <label>Covariates (optional):</label>
                <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">${covCheckboxes}</div>
            </div>
            <div class="aw-form-group">
                <label style="display:flex;align-items:center;gap:8px;text-transform:none;">
                    <input type="checkbox" id="cfg-interactions" checked style="width:auto;">
                    Include interactions
                </label>
            </div>
        `;
    }

    // --- Regression extensions ---
    if (['nominal_logistic', 'ordinal_logistic'].includes(id)) {
        const numericCols = columns.filter(c => c.dtype === 'numeric');
        const checkboxes = numericCols.map(c => `
            <label class="aw-checkbox-item"><input type="checkbox" name="predictors" value="${c.name}"><span>${c.name}</span></label>
        `).join('');
        return `
            <div class="aw-form-group">
                <label>Response (${id === 'ordinal_logistic' ? 'ordered' : 'multi-level'} categories):</label>
                <select id="cfg-response">${colOptions}</select>
            </div>
            <div class="aw-form-group">
                <label>Predictors:</label>
                <div class="aw-checkbox-list" style="max-height:150px;overflow-y:auto;">${checkboxes}</div>
            </div>
        `;
    }
    if (id === 'poisson_regression') {
        const numericCols = columns.filter(c => c.dtype === 'numeric');
        const checkboxes = numericCols.map(c => `
            <label class="aw-checkbox-item"><input type="checkbox" name="predictors" value="${c.name}"><span>${c.name}</span></label>
        `).join('');
        return `
            <div class="aw-form-group">
                <label>Response (counts):</label>
                <select id="cfg-response">${numCols}</select>
            </div>
            <div class="aw-form-group">
                <label>Predictors:</label>
                <div class="aw-checkbox-list" style="max-height:150px;overflow-y:auto;">${checkboxes}</div>
            </div>
            <div class="aw-form-group">
                <label>Offset / exposure column (optional):</label>
                <select id="cfg-offset"><option value="">(none)</option>${numCols}</select>
            </div>
        `;
    }
    if (id === 'orthogonal_regression') {
        return `
            <div class="aw-form-row">
                <div class="aw-form-group">
                    <label>Method X:</label>
                    <select id="cfg-var1">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Method Y:</label>
                    <select id="cfg-var2">${numCols}</select>
                </div>
            </div>
            <div class="aw-form-group">
                <label>Error variance ratio (σ²_x / σ²_y):</label>
                <input type="number" id="cfg-error_ratio" value="1.0" step="0.1" min="0.01">
                <small style="color:#7a8f7a;">1.0 = equal error in both methods (Deming regression)</small>
            </div>
            <div class="aw-form-group">
                <label>Confidence level:</label>
                <select id="cfg-conf">
                    <option value="95">95%</option>
                    <option value="99">99%</option>
                    <option value="90">90%</option>
                </select>
            </div>
        `;
    }
    if (id === 'nonlinear_regression') {
        return `
            <div class="aw-form-row">
                <div class="aw-form-group">
                    <label>Predictor (X):</label>
                    <select id="cfg-var1">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Response (Y):</label>
                    <select id="cfg-var2">${numCols}</select>
                </div>
            </div>
            <div class="aw-form-group">
                <label>Model:</label>
                <select id="cfg-model">
                    <option value="exponential">Exponential: a·e^(bx)</option>
                    <option value="power">Power: a·x^b</option>
                    <option value="logarithmic">Logarithmic: a + b·ln(x)</option>
                    <option value="logistic4">4-Parameter Logistic (S-curve)</option>
                    <option value="gompertz">Gompertz growth</option>
                    <option value="michaelis_menten">Michaelis-Menten: Vmax·x/(Km+x)</option>
                    <option value="gaussian_peak">Gaussian peak</option>
                    <option value="asymptotic">Asymptotic: a·(1-e^(-bx))</option>
                    <option value="polynomial3">Cubic polynomial</option>
                </select>
            </div>
            <div class="aw-form-group">
                <label>Confidence level:</label>
                <select id="cfg-conf">
                    <option value="95">95%</option>
                    <option value="99">99%</option>
                    <option value="90">90%</option>
                </select>
            </div>
        `;
    }

    // --- Acceptance Sampling extensions ---
    if (id === 'variable_acceptance_sampling') {
        return `
            <div class="aw-form-row">
                <div class="aw-form-group">
                    <label>AQL (%):</label>
                    <input type="number" id="cfg-aql" value="1.0" step="0.1" min="0">
                </div>
                <div class="aw-form-group">
                    <label>LTPD / RQL (%):</label>
                    <input type="number" id="cfg-ltpd" value="5.0" step="0.5" min="0">
                </div>
            </div>
            <div class="aw-form-row">
                <div class="aw-form-group">
                    <label>α (producer risk):</label>
                    <input type="number" id="cfg-alpha" value="0.05" step="0.01" min="0" max="1">
                </div>
                <div class="aw-form-group">
                    <label>β (consumer risk):</label>
                    <input type="number" id="cfg-beta" value="0.10" step="0.01" min="0" max="1">
                </div>
            </div>
            <div class="aw-form-group">
                <label>Lot size:</label>
                <input type="number" id="cfg-lot_size" value="1000" step="100" min="1">
            </div>
            <div class="aw-form-group">
                <label>Spec type:</label>
                <select id="cfg-spec_type">
                    <option value="lower">Lower spec (LSL only)</option>
                    <option value="upper">Upper spec (USL only)</option>
                    <option value="both">Both specs</option>
                </select>
            </div>
        `;
    }
    if (id === 'multiple_plan_comparison') {
        return `
            <div class="aw-form-group">
                <label>Define plans to compare:</label>
                <div id="plan-rows">
                    <div class="aw-form-row" style="margin-bottom:4px;">
                        <input type="text" class="plan-name" placeholder="Plan name" value="Plan A" style="flex:1;">
                        <input type="number" class="plan-n" placeholder="n" value="50" style="width:60px;">
                        <input type="number" class="plan-c" placeholder="c" value="2" style="width:60px;">
                    </div>
                    <div class="aw-form-row" style="margin-bottom:4px;">
                        <input type="text" class="plan-name" placeholder="Plan name" value="Plan B" style="flex:1;">
                        <input type="number" class="plan-n" placeholder="n" value="80" style="width:60px;">
                        <input type="number" class="plan-c" placeholder="c" value="3" style="width:60px;">
                    </div>
                </div>
                <button type="button" class="aw-btn-small" onclick="document.getElementById('plan-rows').insertAdjacentHTML('beforeend', '<div class=\\'aw-form-row\\' style=\\'margin-bottom:4px;\\'><input type=\\'text\\' class=\\'plan-name\\' placeholder=\\'Plan name\\' style=\\'flex:1;\\'><input type=\\'number\\' class=\\'plan-n\\' placeholder=\\'n\\' value=\\'50\\' style=\\'width:60px;\\'><input type=\\'number\\' class=\\'plan-c\\' placeholder=\\'c\\' value=\\'2\\' style=\\'width:60px;\\'></div>')">+ Add Plan</button>
            </div>
            <div class="aw-form-row">
                <div class="aw-form-group">
                    <label>Lot size:</label>
                    <input type="number" id="cfg-lot_size" value="1000" step="100">
                </div>
                <div class="aw-form-group">
                    <label>AQL:</label>
                    <input type="number" id="cfg-aql" value="0.01" step="0.005">
                </div>
            </div>
        `;
    }

    // --- MSA extension ---
    if (id === 'gage_rr_expanded') {
        const catCols = columns.filter(c => c.dtype === 'text');
        const factorCheckboxes = catCols.map(c => `
            <label class="aw-checkbox-item"><input type="checkbox" name="factors" value="${c.name}"><span>${c.name}</span></label>
        `).join('') || '<span style="color:#7a8f7a;">No categorical columns</span>';
        return `
            <div class="aw-form-group">
                <label>Measurement:</label>
                <select id="cfg-measurement">${numCols}</select>
            </div>
            <div class="aw-form-group">
                <label>Part:</label>
                <select id="cfg-part">${colOptions}</select>
            </div>
            <div class="aw-form-group">
                <label>Operator:</label>
                <select id="cfg-operator">${colOptions}</select>
            </div>
            <div class="aw-form-group">
                <label>Additional factors:</label>
                <div class="aw-checkbox-list" style="max-height:100px;overflow-y:auto;">${factorCheckboxes}</div>
            </div>
        `;
    }

    // --- SPC extensions ---
    if (id === 'capability_sixpack') {
        return `
            <div class="aw-form-group">
                <label>Variable:</label>
                <select id="cfg-var">${numCols}</select>
            </div>
            <div class="aw-form-row">
                <div class="aw-form-group">
                    <label>LSL:</label>
                    <input type="number" id="cfg-lsl" step="any" placeholder="Lower spec">
                </div>
                <div class="aw-form-group">
                    <label>USL:</label>
                    <input type="number" id="cfg-usl" step="any" placeholder="Upper spec">
                </div>
            </div>
            <div class="aw-form-group">
                <label>Target (optional):</label>
                <input type="number" id="cfg-target" step="any" placeholder="Nominal target">
            </div>
            <div class="aw-form-group">
                <label>Subgroup size:</label>
                <input type="number" id="cfg-subgroup_size" value="1" min="1" max="50">
            </div>
        `;
    }
    if (id === 'laney_p') {
        return `
            <div class="aw-form-group">
                <label>Defectives (count per subgroup):</label>
                <select id="cfg-defectives">${numCols}</select>
            </div>
            <div class="aw-form-group">
                <label>Sample size (n per subgroup):</label>
                <select id="cfg-sample_size">${numCols}</select>
            </div>
        `;
    }
    if (id === 'laney_u') {
        return `
            <div class="aw-form-group">
                <label>Defects (count per unit):</label>
                <select id="cfg-defects">${numCols}</select>
            </div>
            <div class="aw-form-group">
                <label>Units inspected:</label>
                <select id="cfg-units">${numCols}</select>
            </div>
        `;
    }
    if (id === 'between_within') {
        return `
            <div class="aw-form-group">
                <label>Measurement:</label>
                <select id="cfg-measurement">${numCols}</select>
            </div>
            <div class="aw-form-group">
                <label>Subgroup column (optional):</label>
                <select id="cfg-subgroup"><option value="">(auto — sequential)</option>${colOptions}</select>
            </div>
            <div class="aw-form-group">
                <label>Subgroup size:</label>
                <input type="number" id="cfg-subgroup_size" value="5" min="2" max="50">
            </div>
            <div class="aw-form-row">
                <div class="aw-form-group">
                    <label>LSL (optional):</label>
                    <input type="number" id="cfg-lsl" step="any" placeholder="Lower spec">
                </div>
                <div class="aw-form-group">
                    <label>USL (optional):</label>
                    <input type="number" id="cfg-usl" step="any" placeholder="Upper spec">
                </div>
            </div>
        `;
    }
    if (id === 'mewma') {
        const numericCols = columns.filter(c => c.dtype === 'numeric');
        const checkboxes = numericCols.map(c => `
            <label class="aw-checkbox-item"><input type="checkbox" name="variables" value="${c.name}"><span>${c.name}</span></label>
        `).join('');
        return `
            <div class="aw-form-group">
                <label>Variables (select 2+):</label>
                <div class="aw-checkbox-list" style="max-height:150px;overflow-y:auto;">${checkboxes}</div>
                <div style="margin-top:6px;">
                    <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('variables', true)">Select All</button>
                    <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('variables', false)">Clear</button>
                </div>
            </div>
            <div class="aw-form-group">
                <label>Smoothing λ:</label>
                <select id="cfg-lambda">
                    <option value="0.05">0.05 (sensitive)</option>
                    <option value="0.1" selected>0.10 (default)</option>
                    <option value="0.2">0.20</option>
                    <option value="0.3">0.30 (fast response)</option>
                </select>
            </div>
        `;
    }
    if (id === 'generalized_variance') {
        const numericCols = columns.filter(c => c.dtype === 'numeric');
        const checkboxes = numericCols.map(c => `
            <label class="aw-checkbox-item"><input type="checkbox" name="variables" value="${c.name}"><span>${c.name}</span></label>
        `).join('');
        return `
            <div class="aw-form-group">
                <label>Variables (select 2+):</label>
                <div class="aw-checkbox-list" style="max-height:150px;overflow-y:auto;">${checkboxes}</div>
                <div style="margin-top:6px;">
                    <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('variables', true)">Select All</button>
                    <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('variables', false)">Clear</button>
                </div>
            </div>
            <div class="aw-form-group">
                <label>Subgroup size:</label>
                <input type="number" id="cfg-subgroup_size" value="5" min="2" max="50">
            </div>
        `;
    }

    // --- Visualization extensions ---
    if (id === 'dotplot') {
        return `
            <div class="aw-form-group">
                <label>Variable:</label>
                <select id="cfg-var">${numCols}</select>
            </div>
            <div class="aw-form-group">
                <label>Group by (optional):</label>
                <select id="cfg-group"><option value="">(none)</option>${colOptions}</select>
            </div>
        `;
    }
    if (id === 'individual_value_plot') {
        return `
            <div class="aw-form-group">
                <label>Variable:</label>
                <select id="cfg-var">${numCols}</select>
            </div>
            <div class="aw-form-group">
                <label>Group by (optional):</label>
                <select id="cfg-group"><option value="">(none)</option>${colOptions}</select>
            </div>
            <div class="aw-form-group">
                <label style="display:flex;align-items:center;gap:8px;text-transform:none;">
                    <input type="checkbox" id="cfg-show_mean" checked style="width:auto;">
                    Show mean line
                </label>
            </div>
        `;
    }
    if (id === 'interval_plot') {
        return `
            <div class="aw-form-group">
                <label>Variable:</label>
                <select id="cfg-var">${numCols}</select>
            </div>
            <div class="aw-form-group">
                <label>Group by:</label>
                <select id="cfg-group">${colOptions}</select>
            </div>
            <div class="aw-form-group">
                <label>Confidence level:</label>
                <select id="cfg-confidence">
                    <option value="0.95">95%</option>
                    <option value="0.99">99%</option>
                    <option value="0.90">90%</option>
                </select>
            </div>
        `;
    }
    if (id === 'contour' || id === 'contour_overlay') {
        const isOverlay = id === 'contour_overlay';
        if (isOverlay) {
            const numericCols = columns.filter(c => c.dtype === 'numeric');
            const checkboxes = numericCols.map(c => `
                <label class="aw-checkbox-item"><input type="checkbox" name="z_columns" value="${c.name}"><span>${c.name}</span></label>
            `).join('');
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>X (factor 1):</label>
                        <select id="cfg-x">${numCols}</select>
                    </div>
                    <div class="aw-form-group">
                        <label>Y (factor 2):</label>
                        <select id="cfg-y">${numCols}</select>
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Response columns (select 2+):</label>
                    <div class="aw-checkbox-list" style="max-height:150px;overflow-y:auto;">${checkboxes}</div>
                </div>
            `;
        }
        return `
            <div class="aw-form-row">
                <div class="aw-form-group">
                    <label>X:</label>
                    <select id="cfg-x">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Y:</label>
                    <select id="cfg-y">${numCols}</select>
                </div>
            </div>
            <div class="aw-form-group">
                <label>Z (response):</label>
                <select id="cfg-z">${numCols}</select>
            </div>
        `;
    }
    if (id === 'surface_3d') {
        return `
            <div class="aw-form-row">
                <div class="aw-form-group">
                    <label>X:</label>
                    <select id="cfg-x">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Y:</label>
                    <select id="cfg-y">${numCols}</select>
                </div>
            </div>
            <div class="aw-form-group">
                <label>Z (response):</label>
                <select id="cfg-z">${numCols}</select>
            </div>
        `;
    }
    if (id === 'run_chart') {
        return `
            <div class="aw-form-group">
                <label>Variable:</label>
                <select id="cfg-var">${numCols}</select>
            </div>
            <div class="aw-form-group">
                <label>Time / order column (optional):</label>
                <select id="cfg-time_col"><option value="">(use row order)</option>${colOptions}</select>
            </div>
        `;
    }

    // --- ML / Multivariate extensions ---
    if (id === 'factor_analysis') {
        const numericCols = columns.filter(c => c.dtype === 'numeric');
        const checkboxes = numericCols.map(c => `
            <label class="aw-checkbox-item"><input type="checkbox" name="variables" value="${c.name}" checked><span>${c.name}</span></label>
        `).join('');
        return `
            <div class="aw-form-group">
                <label>Variables:</label>
                <div class="aw-checkbox-list" style="max-height:150px;overflow-y:auto;">${checkboxes}</div>
                <div style="margin-top:6px;">
                    <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('variables', true)">Select All</button>
                    <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('variables', false)">Clear</button>
                </div>
            </div>
            <div class="aw-form-row">
                <div class="aw-form-group">
                    <label>Number of factors:</label>
                    <input type="number" id="cfg-n_factors" value="" placeholder="Auto (eigenvalue > 1)" min="1" max="20">
                </div>
                <div class="aw-form-group">
                    <label>Rotation:</label>
                    <select id="cfg-rotation">
                        <option value="varimax">Varimax (orthogonal)</option>
                        <option value="promax">Promax (oblique)</option>
                        <option value="none">None</option>
                    </select>
                </div>
            </div>
        `;
    }
    if (id === 'correspondence_analysis') {
        const catCols = columns.filter(c => c.dtype === 'text').map(c => `<option value="${c.name}">${c.name}</option>`).join('');
        return `
            <div class="aw-form-group">
                <label>Row variable:</label>
                <select id="cfg-row_var">${catCols || colOptions}</select>
            </div>
            <div class="aw-form-group">
                <label>Column variable:</label>
                <select id="cfg-col_var">${catCols || colOptions}</select>
            </div>
        `;
    }
    if (id === 'item_analysis') {
        const numericCols = columns.filter(c => c.dtype === 'numeric');
        const checkboxes = numericCols.map(c => `
            <label class="aw-checkbox-item"><input type="checkbox" name="items" value="${c.name}" checked><span>${c.name}</span></label>
        `).join('');
        return `
            <div class="aw-form-group">
                <label>Scale items (select all items in the scale):</label>
                <div class="aw-checkbox-list" style="max-height:200px;overflow-y:auto;">${checkboxes}</div>
                <div style="margin-top:6px;">
                    <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('items', true)">Select All</button>
                    <button type="button" class="aw-btn-small" onclick="toggleAllCheckboxes('items', false)">Clear</button>
                </div>
            </div>
            <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                <strong style="color:var(--aw-accent);">Cronbach's α:</strong> Measures internal consistency. α ≥ 0.70 generally acceptable; ≥ 0.80 good.
            </div>
        `;
    }

    // --- Tools extensions ---
    if (id === 'box_cox' || id === 'johnson_transform') {
        return `
            <div class="aw-form-group">
                <label>Variable:</label>
                <select id="cfg-var">${numCols}</select>
            </div>
            <div style="background:rgba(74,159,110,0.1);padding:8px;border-radius:4px;font-size:10px;color:var(--aw-text-muted);">
                <strong style="color:var(--aw-accent);">${id === 'box_cox' ? 'Box-Cox' : 'Johnson'}:</strong> ${id === 'box_cox' ? 'Finds optimal λ to transform data toward normality. Requires positive data.' : 'Selects best SB/SL/SU family to transform data to normality. Works with any data range.'}
            </div>
        `;
    }
    if (id === 'grubbs_test') {
        return `
            <div class="aw-form-group">
                <label>Variable:</label>
                <select id="cfg-var">${numCols}</select>
            </div>
            <div class="aw-form-group">
                <label>Significance level:</label>
                <select id="cfg-alpha">
                    <option value="0.05">0.05 (95% confidence)</option>
                    <option value="0.01">0.01 (99% confidence)</option>
                    <option value="0.10">0.10 (90% confidence)</option>
                </select>
            </div>
        `;
    }
    if (id === 'graphical_summary') {
        return `
            <div class="aw-form-group">
                <label>Variable:</label>
                <select id="cfg-var">${numCols}</select>
            </div>
            <div class="aw-form-group">
                <label>Confidence level:</label>
                <select id="cfg-confidence">
                    <option value="0.95">95%</option>
                    <option value="0.99">99%</option>
                    <option value="0.90">90%</option>
                </select>
            </div>
        `;
    }
    if (id === 'auto_profile') {
        return `
            <div style="background:rgba(74,159,110,0.1);padding:12px;border-radius:4px;font-size:11px;color:var(--aw-text-muted);">
                <strong style="color:var(--aw-accent);">Auto Profile</strong> analyzes all columns in your dataset automatically.
                No configuration needed — click Execute to generate the overview.
            </div>
        `;
    }

    if (type === 'siop') {
        if (id === 'abc_analysis') {
            return `
                <div class="aw-form-group">
                    <label>Item/SKU column:</label>
                    <select id="cfg-item_col">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Value column (annual $ or revenue):</label>
                    <select id="cfg-value_col"><option value="">Auto-detect</option>${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>XYZ: Period demand columns (multi-select for CoV):</label>
                    <div class="aw-checkbox-list" style="max-height:120px;overflow-y:auto;">
                        ${columns.filter(c => c.dtype === 'numeric').map(c => `
                            <label class="aw-checkbox-item">
                                <input type="checkbox" name="demand_cols" value="${c.name}">
                                <span>${c.name}</span>
                            </label>
                        `).join('')}
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>A threshold (%):</label>
                        <input type="number" id="cfg-a_threshold" value="80" min="50" max="95">
                    </div>
                    <div class="aw-form-group">
                        <label>B threshold (%):</label>
                        <input type="number" id="cfg-b_threshold" value="95" min="80" max="99">
                    </div>
                </div>
            `;
        }
        if (id === 'eoq') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Annual demand (D):</label>
                        <input type="number" id="cfg-demand" value="10000" min="1">
                    </div>
                    <div class="aw-form-group">
                        <label>Order cost ($/order):</label>
                        <input type="number" id="cfg-order_cost" value="50" min="0.01" step="0.01">
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Unit cost ($):</label>
                        <input type="number" id="cfg-unit_cost" value="25" min="0.01" step="0.01">
                    </div>
                    <div class="aw-form-group">
                        <label>Holding cost (% of unit):</label>
                        <input type="number" id="cfg-holding_pct" value="25" min="1" max="100">
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Or use demand data column:</label>
                    <select id="cfg-demand_col"><option value="">Manual entry above</option>${numCols}</select>
                </div>
            `;
        }
        if (id === 'safety_stock') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Mean demand/period:</label>
                        <input type="number" id="cfg-demand_mean" value="100" min="0" step="0.1">
                    </div>
                    <div class="aw-form-group">
                        <label>Demand std dev:</label>
                        <input type="number" id="cfg-demand_std" value="20" min="0" step="0.1">
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Lead time (periods):</label>
                        <input type="number" id="cfg-lead_time" value="5" min="0.1" step="0.1">
                    </div>
                    <div class="aw-form-group">
                        <label>Lead time std dev:</label>
                        <input type="number" id="cfg-lead_time_std" value="1" min="0" step="0.1">
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Service level (%):</label>
                    <input type="number" id="cfg-service_level" value="95" min="80" max="99.9" step="0.5">
                </div>
                <div class="aw-form-group">
                    <label>Or use demand data column:</label>
                    <select id="cfg-demand_col"><option value="">Manual entry above</option>${numCols}</select>
                </div>
            `;
        }
        if (id === 'inventory_turns') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Annual COGS ($):</label>
                        <input type="number" id="cfg-cogs" value="5000000" min="1">
                    </div>
                    <div class="aw-form-group">
                        <label>Avg inventory ($):</label>
                        <input type="number" id="cfg-avg_inventory" value="800000" min="1">
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Industry benchmark:</label>
                    <select id="cfg-industry">
                        <option value="manufacturing">Manufacturing</option>
                        <option value="retail">Retail</option>
                        <option value="food">Food & Beverage</option>
                        <option value="automotive">Automotive</option>
                        <option value="pharma">Pharmaceutical</option>
                        <option value="electronics">Electronics</option>
                    </select>
                </div>
                <div class="aw-form-group">
                    <label>Or use data columns for trend:</label>
                    <div class="aw-form-row">
                        <select id="cfg-cogs_col"><option value="">None</option>${numCols}</select>
                        <select id="cfg-inventory_col"><option value="">None</option>${numCols}</select>
                    </div>
                </div>
            `;
        }
        if (id === 'service_level') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Mean demand/period:</label>
                        <input type="number" id="cfg-demand_mean" value="100" min="0" step="0.1">
                    </div>
                    <div class="aw-form-group">
                        <label>Demand std dev:</label>
                        <input type="number" id="cfg-demand_std" value="20" min="0" step="0.1">
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Lead time:</label>
                        <input type="number" id="cfg-lead_time" value="5" min="0.1" step="0.1">
                    </div>
                    <div class="aw-form-group">
                        <label>Unit cost ($):</label>
                        <input type="number" id="cfg-unit_cost" value="25" min="0.01" step="0.01">
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Stockout cost ($/unit short, optional):</label>
                    <input type="number" id="cfg-stockout_cost" value="" placeholder="Leave blank for cost-only view" min="0" step="1">
                </div>
            `;
        }
        if (id === 'demand_profile') {
            return `
                <div class="aw-form-group">
                    <label>Demand column:</label>
                    <select id="cfg-demand_col">${numCols}</select>
                </div>
            `;
        }
        if (id === 'kanban_sizing') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Demand/period:</label>
                        <input type="number" id="cfg-demand" value="50" min="1">
                    </div>
                    <div class="aw-form-group">
                        <label>Lead time (periods):</label>
                        <input type="number" id="cfg-lead_time" value="3" min="0.1" step="0.1">
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Safety factor (%):</label>
                        <input type="number" id="cfg-safety_pct" value="15" min="0" max="100">
                    </div>
                    <div class="aw-form-group">
                        <label>Container size:</label>
                        <input type="number" id="cfg-container_size" value="25" min="1">
                    </div>
                </div>
            `;
        }
        if (id === 'epei') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Available hours/day:</label>
                        <input type="number" id="cfg-available_hours" value="8" min="1" max="24" step="0.5">
                    </div>
                    <div class="aw-form-group">
                        <label>Number of parts:</label>
                        <input type="number" id="cfg-num_parts" value="12" min="1">
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Avg changeover (min):</label>
                        <input type="number" id="cfg-changeover_time" value="15" min="1">
                    </div>
                    <div class="aw-form-group">
                        <label>C/O budget (%):</label>
                        <input type="number" id="cfg-target_pct" value="10" min="1" max="50">
                    </div>
                </div>
            `;
        }
        if (id === 'rop_simulation') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Mean demand/period:</label>
                        <input type="number" id="cfg-demand_mean" value="100" min="0">
                    </div>
                    <div class="aw-form-group">
                        <label>Demand std dev:</label>
                        <input type="number" id="cfg-demand_std" value="20" min="0">
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Lead time:</label>
                        <input type="number" id="cfg-lead_time" value="5" min="1">
                    </div>
                    <div class="aw-form-group">
                        <label>LT std dev:</label>
                        <input type="number" id="cfg-lead_time_std" value="1" min="0">
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Reorder point (s):</label>
                        <input type="number" id="cfg-reorder_point" value="680" min="0">
                    </div>
                    <div class="aw-form-group">
                        <label>Order quantity (Q):</label>
                        <input type="number" id="cfg-order_quantity" value="400" min="1">
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Holding $/unit/period:</label>
                        <input type="number" id="cfg-holding_cost" value="0.02" min="0" step="0.01">
                    </div>
                    <div class="aw-form-group">
                        <label>Stockout $/unit:</label>
                        <input type="number" id="cfg-stockout_cost" value="5" min="0" step="0.1">
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Simulation runs:</label>
                    <input type="number" id="cfg-runs" value="1000" min="100" max="5000">
                </div>
            `;
        }
        if (id === 'mrp_netting') {
            return `
                <div class="aw-form-group">
                    <label>Gross requirements column:</label>
                    <select id="cfg-gross_col">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Scheduled receipts column (optional):</label>
                    <select id="cfg-receipts_col"><option value="">None</option>${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>On-hand inventory:</label>
                        <input type="number" id="cfg-on_hand" value="0" min="0">
                    </div>
                    <div class="aw-form-group">
                        <label>Safety stock:</label>
                        <input type="number" id="cfg-safety_stock" value="0" min="0">
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Lead time (periods):</label>
                        <input type="number" id="cfg-lead_time" value="2" min="1">
                    </div>
                    <div class="aw-form-group">
                        <label>Lot sizing:</label>
                        <select id="cfg-lot_size">
                            <option value="lot_for_lot">Lot-for-Lot</option>
                            <option value="fixed">Fixed Quantity</option>
                            <option value="eoq">EOQ</option>
                        </select>
                    </div>
                </div>
            `;
        }
        if (id === 'inventory_policy_wizard') {
            return `
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Value/revenue column:</label>
                        <select id="cfg-value_col">${numCols}</select>
                    </div>
                    <div class="aw-form-group">
                        <label>Demand column:</label>
                        <select id="cfg-demand_col">${numCols}</select>
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Demand std dev column (optional):</label>
                        <select id="cfg-demand_std_col"><option value="">Auto (20% CV)</option>${numCols}</select>
                    </div>
                    <div class="aw-form-group">
                        <label>Lead time (periods):</label>
                        <input type="number" id="cfg-lead_time" value="14" min="1" step="1">
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Container size:</label>
                        <input type="number" id="cfg-container_size" value="1" min="1">
                    </div>
                    <div class="aw-form-group">
                        <label>Order cost ($):</label>
                        <input type="number" id="cfg-order_cost" value="100" min="0" step="10">
                    </div>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>Class A service level:</label>
                        <input type="number" id="cfg-a_service" value="0.99" min="0.5" max="1" step="0.01">
                    </div>
                    <div class="aw-form-group">
                        <label>Class B service level:</label>
                        <input type="number" id="cfg-b_service" value="0.95" min="0.5" max="1" step="0.01">
                    </div>
                    <div class="aw-form-group">
                        <label>Class C service level:</label>
                        <input type="number" id="cfg-c_service" value="0.90" min="0.5" max="1" step="0.01">
                    </div>
                </div>
            `;
        }
        return `<p style="color:#9aaa9a;font-size:11px;">Select an S&OP analysis.</p>`;
    }

    // Robust — D-Type, PBS, Conformal, Causal, Quality Economics
    if (type === 'robust') {
        // D-Type analyses need measurement + factor columns
        if (id.startsWith('d_')) {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Factor / batch column:</label>
                    <select id="cfg-factor"><option value="">None</option>${colOptions}</select>
                </div>
                ${id === 'd_cpk' || id === 'd_nonnorm' || id === 'd_multi' ? `
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>LSL:</label>
                        <input type="number" id="cfg-lsl" step="any" placeholder="Lower spec">
                    </div>
                    <div class="aw-form-group">
                        <label>USL:</label>
                        <input type="number" id="cfg-usl" step="any" placeholder="Upper spec">
                    </div>
                </div>` : ''}
            `;
        }
        // PBS analyses — mode-specific config matching backend pbs_engine.py
        if (id.startsWith('pbs_')) {
            const mode = id.replace('pbs_', '');
            const needsSpec = ['full','predictive','cpk','cpk_traj','health','edetector'].includes(mode);
            const needsTarget = ['full','belief','evidence','adaptive','edetector'].includes(mode);
            const needsHazard = ['full','belief','health'].includes(mode);
            const needsBeta = ['full','belief','health'].includes(mode);
            const descs = {
                full: 'All charts: belief, e-detector, evidence, adaptive limits, predictive, Cpk, health gauge.',
                belief: 'BOCPD — posterior probability of process shift at each time step.',
                edetector: 'CUSUM e-detector — distribution-free changepoint detection with guaranteed ARL.',
                evidence: 'Anytime-valid e-values accumulate evidence against the in-control hypothesis.',
                predictive: 'Bayesian linear trend on rolling window with prediction fan.',
                adaptive: 'Control limits from posterior predictive Student-t. Narrows as data arrives.',
                cpk: 'Full posterior of Cpk via ancestral sampling from Normal-Gamma.',
                cpk_traj: 'Rolling Bayesian Cpk over time with trend detection.',
                health: 'Log-linear opinion pool fusing stability, capability, and trend.'
            };
            return `
                <p style="color:var(--aw-text-muted);font-size:10px;margin-bottom:8px;">${descs[mode] || ''}</p>
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-column">${numCols}</select>
                </div>
                ${needsSpec ? `<div class="aw-form-row">
                    <div class="aw-form-group"><label>LSL:</label><input type="number" id="cfg-LSL" step="any" placeholder="Lower spec"></div>
                    <div class="aw-form-group"><label>USL:</label><input type="number" id="cfg-USL" step="any" placeholder="Upper spec"></div>
                </div>` : ''}
                ${needsTarget ? `<div class="aw-form-group">
                    <label>Target (optional):</label>
                    <input type="number" id="cfg-target" step="any" placeholder="Process target">
                </div>` : ''}
                ${needsHazard ? `<div class="aw-form-group">
                    <label>Hazard \u03bb:</label>
                    <input type="number" id="cfg-hazard_lambda" step="1" placeholder="Auto (n/4)" min="10" max="10000">
                    <small style="color:var(--aw-text-muted);font-size:9px;">Expected regime length — leave blank for auto</small>
                </div>` : ''}
                ${needsBeta ? `<div class="aw-form-group">
                    <label>Robustness \u03b2: <span id="pbs-beta-val">0.0</span></label>
                    <input type="range" id="cfg-beta_robustness" min="0" max="0.5" step="0.05" value="0" oninput="document.getElementById('pbs-beta-val').textContent=this.value">
                    <small style="color:var(--aw-text-muted);font-size:9px;">0 = standard BOCPD, 0.1–0.3 = robust</small>
                </div>` : ''}
                ${mode === 'full' ? `<div class="aw-form-group">
                    <label>Taguchi k (optional):</label>
                    <input type="number" id="cfg-taguchi_k" step="any" min="0" placeholder="Auto from specs">
                    <small style="color:var(--aw-text-muted);font-size:9px;">Defaults to 1/\u0394\u2080\u00b2 when specs set</small>
                </div>` : ''}
                ${mode === 'edetector' ? `<div class="aw-form-group">
                    <label>\u03b1 (false alarm rate):</label>
                    <input type="number" id="cfg-edetector_alpha" step="0.01" value="0.05" min="0.001" max="0.5">
                    <small style="color:var(--aw-text-muted);font-size:9px;">Guaranteed: ARL \u2265 1/\u03b1</small>
                </div>` : ''}
                ${['cpk','cpk_traj'].includes(mode) ? `<p style="color:var(--aw-warning);font-size:9px;margin-top:4px;">Both USL and LSL required for Cpk</p>` : ''}
            `;
        }
        // Conformal control chart
        if (id === 'conformal_control') {
            return `
                <div class="aw-form-group">
                    <label>Measurement Column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Chart Type:</label>
                    <select id="cfg-chart_type">
                        <option value="individuals">Individual Values</option>
                        <option value="subgroup_mean">Subgroup Mean</option>
                        <option value="subgroup_range">Subgroup Range</option>
                    </select>
                </div>
                <div class="aw-form-group">
                    <label>Subgroup Size:</label>
                    <input type="number" id="cfg-subgroup_size" value="5" min="2" max="25">
                </div>
                <div class="aw-form-group">
                    <label>False Alarm Rate (\u03b1):</label>
                    <input type="number" id="cfg-alpha" step="0.01" value="0.05" min="0.001" max="0.5">
                </div>
                <div class="aw-form-group">
                    <label>Calibration Fraction (Phase I split):</label>
                    <input type="number" id="cfg-calibration_fraction" step="0.05" value="0.5" min="0.1" max="0.9">
                </div>
                <div class="aw-form-group">
                    <label>Spike Threshold (\u00d7 median width):</label>
                    <input type="number" id="cfg-spike_threshold" step="0.5" value="2.0" min="1" max="5">
                </div>
                <p style="color:var(--aw-text-muted);font-size:9px;margin-top:4px;">Distribution-free control chart with guaranteed false alarm rate. No normality assumption. <em>Burger et al. (2025)</em></p>
            `;
        }
        // Conformal monitor (multivariate)
        if (id === 'conformal_monitor') {
            return `
                <div class="aw-form-group">
                    <label>Variables (select 2+):</label>
                    <select id="cfg-variables" multiple size="5" style="min-height:100px;">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Anomaly Model:</label>
                    <select id="cfg-model">
                        <option value="isolation_forest">Isolation Forest (default)</option>
                        <option value="mahalanobis">Mahalanobis Distance</option>
                    </select>
                </div>
                <div class="aw-form-group">
                    <label>False Alarm Rate (\u03b1):</label>
                    <input type="number" id="cfg-alpha" step="0.01" value="0.05" min="0.001" max="0.5">
                </div>
                <div class="aw-form-group">
                    <label>Calibration Fraction (Phase I split):</label>
                    <input type="number" id="cfg-calibration_fraction" step="0.05" value="0.5" min="0.1" max="0.9">
                </div>
                <p style="color:var(--aw-text-muted);font-size:9px;margin-top:4px;">Multivariate anomaly detection with conformal p-values. Guaranteed false alarm rate. <em>Burger et al. (2025)</em></p>
            `;
        }
        // Drift, anytime
        if (id === 'drift' || id.startsWith('anytime_')) {
            return `
                <div class="aw-form-group">
                    <label>Variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Group / time column (optional):</label>
                    <select id="cfg-group"><option value="">None</option>${colOptions}</select>
                </div>
            `;
        }
        if (id.startsWith('causal_') || id === 'ishap') {
            const checkboxes = columns.filter(c => c.dtype === 'numeric').map(c => `
                <label class="aw-checkbox-item">
                    <input type="checkbox" name="vars" value="${c.name}" checked>
                    <span>${c.name}</span>
                </label>
            `).join('');
            return `
                <div class="aw-form-group">
                    <label>Variables:</label>
                    <div class="aw-checkbox-list" style="max-height:200px;overflow-y:auto;">${checkboxes}</div>
                </div>
            `;
        }
        // Quality economics
        if (id === 'taguchi' || id === 'process_decision' || id === 'lot_sentencing' || id === 'cost_of_quality') {
            return `
                <div class="aw-form-group">
                    <label>Measurement column:</label>
                    <select id="cfg-measurement">${numCols}</select>
                </div>
                <div class="aw-form-row">
                    <div class="aw-form-group">
                        <label>LSL:</label>
                        <input type="number" id="cfg-lsl" step="any" placeholder="Lower spec">
                    </div>
                    <div class="aw-form-group">
                        <label>USL:</label>
                        <input type="number" id="cfg-usl" step="any" placeholder="Upper spec">
                    </div>
                </div>
                <div class="aw-form-group">
                    <label>Target:</label>
                    <input type="number" id="cfg-target" step="any" placeholder="Nominal value">
                </div>
            `;
        }
        return `<p style="color:#9aaa9a;font-size:11px;">Select a robust analysis.</p>`;
    }

    // DOE
    if (type === 'doe') {
        if (id === 'main_effects' || id === 'interaction' || id === 'doe_contour' || id === 'doe_optimize') {
            return `
                <div class="aw-form-group">
                    <label>Response column:</label>
                    <select id="cfg-response">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Factor columns:</label>
                    <select id="cfg-factors" multiple style="height:100px;">${colOptions}</select>
                </div>
            `;
        }
        if (id === 'power') {
            return `
                <div class="aw-form-group">
                    <label>Number of factors:</label>
                    <input type="number" id="cfg-factors" value="3" min="2" max="15">
                </div>
                <div class="aw-form-group">
                    <label>Desired power:</label>
                    <select id="cfg-power">
                        <option value="0.8">80%</option>
                        <option value="0.9" selected>90%</option>
                        <option value="0.95">95%</option>
                    </select>
                </div>
                <div class="aw-form-group">
                    <label>Effect size (StDev units):</label>
                    <input type="number" id="cfg-effect_size" value="1.0" step="0.1">
                </div>
            `;
        }
        // Design creation — factor builder (name + low + high levels)
        return `
            <div class="aw-form-group">
                <label>Factors:</label>
                <div id="doe-factor-rows">
                    <div class="doe-factor-row" style="display:flex;gap:6px;margin-bottom:4px;">
                        <input type="text" class="doe-fname" placeholder="Name" value="Factor A" style="flex:2;">
                        <input type="text" class="doe-flow" placeholder="Low" value="-1" style="flex:1;">
                        <input type="text" class="doe-fhigh" placeholder="High" value="1" style="flex:1;">
                    </div>
                    <div class="doe-factor-row" style="display:flex;gap:6px;margin-bottom:4px;">
                        <input type="text" class="doe-fname" placeholder="Name" value="Factor B" style="flex:2;">
                        <input type="text" class="doe-flow" placeholder="Low" value="-1" style="flex:1;">
                        <input type="text" class="doe-fhigh" placeholder="High" value="1" style="flex:1;">
                    </div>
                </div>
                <div style="display:flex;gap:6px;margin-top:4px;">
                    <button type="button" class="aw-btn-sm" onclick="addDOEFactor()" style="font-size:11px;padding:2px 8px;background:var(--aw-accent);color:white;border:none;border-radius:3px;cursor:pointer;">+ Factor</button>
                    <button type="button" class="aw-btn-sm" onclick="removeDOEFactor()" style="font-size:11px;padding:2px 8px;background:transparent;color:var(--aw-text-muted);border:1px solid var(--aw-border);border-radius:3px;cursor:pointer;">\u2212 Factor</button>
                </div>
            </div>
            <div class="aw-form-row" style="display:flex;gap:8px;">
                <div class="aw-form-group" style="flex:1;">
                    <label>Replicates:</label>
                    <input type="number" id="cfg-replicates" value="1" min="1" max="10">
                </div>
                <div class="aw-form-group" style="flex:1;">
                    <label>Center points:</label>
                    <input type="number" id="cfg-center_points" value="0" min="0" max="10">
                </div>
            </div>
            <div class="aw-form-group">
                <label>Response name:</label>
                <input type="text" id="cfg-response_name" value="Response" placeholder="e.g. Yield, Strength">
            </div>
        `;
    }

    // Prepare — data prep tools
    if (type === 'prepare') {
        if (id === 'outliers') {
            return `
                <div class="aw-form-group">
                    <label>Variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Method:</label>
                    <select id="cfg-method">
                        <option value="iqr">IQR (1.5×)</option>
                        <option value="zscore">Z-score (3σ)</option>
                        <option value="mad">MAD</option>
                        <option value="mahalanobis">Mahalanobis</option>
                    </select>
                </div>
            `;
        }
        if (id === 'encode') {
            return `
                <div class="aw-form-group">
                    <label>Column to encode:</label>
                    <select id="cfg-var">${colOptions}</select>
                </div>
                <div class="aw-form-group">
                    <label>Method:</label>
                    <select id="cfg-method">
                        <option value="onehot">One-Hot</option>
                        <option value="label">Label Encoding</option>
                    </select>
                </div>
            `;
        }
        if (id === 'scale') {
            return `
                <div class="aw-form-group">
                    <label>Variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Method:</label>
                    <select id="cfg-method">
                        <option value="zscore">Z-score</option>
                        <option value="minmax">Min-Max (0–1)</option>
                        <option value="robust">Robust (median/IQR)</option>
                    </select>
                </div>
            `;
        }
        if (id === 'bin') {
            return `
                <div class="aw-form-group">
                    <label>Variable:</label>
                    <select id="cfg-var">${numCols}</select>
                </div>
                <div class="aw-form-group">
                    <label>Number of bins:</label>
                    <input type="number" id="cfg-bins" value="5" min="2" max="50">
                </div>
                <div class="aw-form-group">
                    <label>Method:</label>
                    <select id="cfg-method">
                        <option value="equal_width">Equal Width</option>
                        <option value="equal_freq">Equal Frequency</option>
                    </select>
                </div>
            `;
        }
        // Triage, profile, missing, duplicates, meta, effect_size — use variable picker
        return `
            <div class="aw-form-group">
                <label>Variable (optional):</label>
                <select id="cfg-var"><option value="">All columns</option>${numCols}</select>
            </div>
        `;
    }

    return '<p style="color:#9aaa9a;font-size:11px;">Configure analysis options.</p>';
}

async function executeAnalysis() {
    const noDataTypes = ['simulation', 'bayesian', 'siop', 'doe'];
    if (!currentAnalysis || (!currentData && !noDataTypes.includes(currentAnalysis.type))) {
        alert('Please load data first');
        return;
    }

    // Save form values before closing dialog (for "remember last config")
    try {
        const savedVals = {};
        document.querySelectorAll('#config-dialog-body input[id], #config-dialog-body select').forEach(el => {
            if (el.id) savedVals[el.id] = el.type === 'checkbox' ? el.checked : el.value;
        });
        localStorage.setItem('awLastConfig_' + currentAnalysis.id, JSON.stringify(savedVals));
    } catch (e) { /* ignore quota errors */ }

    closeDialog('config-dialog');

    const config = {};

    // Collect from inputs and selects with IDs
    document.querySelectorAll('#config-dialog-body input[id], #config-dialog-body select').forEach(el => {
        const key = el.id.replace('cfg-', '');
        if (el.multiple) {
            config[key] = Array.from(el.selectedOptions).map(o => o.value);
        } else if (el.type === 'checkbox' && el.id) {
            config[key] = el.checked;
        } else if (el.type !== 'checkbox') {
            config[key] = el.value;
        }
    });

    // Collect checkboxes by name (for predictors, features, etc.)
    const checkboxGroups = {};
    document.querySelectorAll('#config-dialog-body input[type="checkbox"]:checked').forEach(cb => {
        const name = cb.name;
        if (!checkboxGroups[name]) checkboxGroups[name] = [];
        checkboxGroups[name].push(cb.value);
    });
    Object.assign(config, checkboxGroups);

    // Collect plan rows for multiple_plan_comparison
    if (currentAnalysis.id === 'multiple_plan_comparison') {
        const plans = [];
        document.querySelectorAll('#plan-rows .aw-form-row').forEach(row => {
            const name = row.querySelector('.plan-name')?.value || 'Plan';
            const n = parseInt(row.querySelector('.plan-n')?.value) || 50;
            const c = parseInt(row.querySelector('.plan-c')?.value) || 2;
            plans.push({ name, n, sample_size: n, c, accept_number: c, type: 'single' });
        });
        config.plans = plans;
    }

    const menu = analysisMenus[currentAnalysis.type];
    const item = menu?.items.find(i => i.id === currentAnalysis.id);

    // Add separator before new analysis
    appendSession('separator');
    appendSession('cmd', item?.name || currentAnalysis.id);

    // Handle checkbox for new_tab
    const newTabCheckbox = document.getElementById('cfg-new_tab');
    config.new_tab = newTabCheckbox ? newTabCheckbox.checked : false;

    try {
        // Data tools use a different endpoint
        if (currentAnalysis.type === 'tools') {
            await executeDataTool(currentAnalysis.id, config, item?.name);
            return;
        }

        // === Type remapping ===
        // Menu types (robust, doe, prepare) bundle analyses from multiple backend types.
        // Remap to the correct backend analysis_type before POSTing.
        const _TYPE_REMAP = {
            // robust menu → backend types
            'd_chart': 'd_type', 'd_cpk': 'd_type', 'd_nonnorm': 'd_type',
            'd_equiv': 'd_type', 'd_sig': 'd_type', 'd_multi': 'd_type',
            'pbs_full': 'pbs', 'pbs_belief': 'pbs', 'pbs_evidence': 'pbs',
            'pbs_edetector': 'pbs', 'pbs_adaptive': 'pbs', 'pbs_predictive': 'pbs',
            'pbs_cpk': 'pbs', 'pbs_cpk_traj': 'pbs', 'pbs_health': 'pbs',
            'conformal_control': 'spc', 'conformal_monitor': 'spc',
            'drift': 'drift',
            'anytime_ab': 'anytime', 'anytime_onesample': 'anytime',
            'causal_pc': 'causal', 'causal_lingam': 'causal',
            'ishap': 'ishap',
            'taguchi': 'quality_econ', 'process_decision': 'quality_econ',
            'lot_sentencing': 'quality_econ', 'cost_of_quality': 'quality_econ',
            // bayesian menu → reroute items that live in other backend types
            'bayes_spc_capability': 'viz', 'bayes_spc_changepoint': 'viz',
            'bayes_spc_control': 'viz', 'bayes_spc_acceptance': 'viz',
            'bayes_msa': 'bayes_msa',
            // prepare menu → stats backend (with ID normalization)
            'auto_profile': 'stats', 'graphical_summary': 'stats', 'run_chart': 'stats',
            'missing': 'stats', 'duplicates': 'stats', 'outliers': 'stats',
            'encode': 'stats', 'scale': 'stats', 'bin': 'stats',
            'meta_analysis': 'stats', 'effect_size': 'stats',
        };
        // Normalize analysis IDs that differ between menu and backend
        const _ID_REMAP = {
            'missing': 'missing_data_analysis',
            'duplicates': 'duplicate_analysis',
            'outliers': 'outlier_analysis',
            'effect_size': 'effect_size_calculator',
            'taguchi': 'taguchi_loss',
            'drift': 'drift_report',
            // bayesian menu IDs → backend registry IDs
            'bayes_cpk_predict': 'bayes_capability_prediction',
            'bayes_reliability_demo': 'bayes_demo',
            'bayes_system_reliability': 'bayes_system',
            'bayes_competing_risks': 'bayes_comprisk',
        };

        let backendType = currentAnalysis.type;
        let backendId = currentAnalysis.id;

        // Apply remaps for menu types that don't match backend types
        if (['robust', 'prepare', 'bayesian'].includes(backendType)) {
            backendType = _TYPE_REMAP[backendId] || backendType;
        }
        if (_ID_REMAP[backendId]) {
            backendId = _ID_REMAP[backendId];
        }

        // DOE: design creation uses /api/experimenter/design/, analysis uses /api/experimenter/analyze/
        const _DOE_DESIGNS = ['full_factorial', 'fractional_factorial', 'plackett_burman',
            'ccd', 'taguchi_design', 'latin_square', 'd_optimal'];
        if (currentAnalysis.type === 'doe') {
            if (_DOE_DESIGNS.includes(currentAnalysis.id)) {
                await executeDOEDesign(currentAnalysis.id, config, item?.name);
                return;
            } else if (currentAnalysis.id === 'power') {
                await executeDOEPower(config, item?.name);
                return;
            } else if (['main_effects', 'interaction', 'doe_contour', 'doe_optimize'].includes(currentAnalysis.id)) {
                await executeDOEAnalysis(currentAnalysis.id, config, item?.name);
                return;
            }
        }

        // Triage uses its own endpoint
        if (backendId === 'triage') {
            window.location.href = '/app/triage/';
            return;
        }

        const response = await fetch('/api/analysis/run/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify({
                type: backendType,
                analysis: backendId,
                config: config,
                data_id: currentData?.id,
                notebook_id: currentNotebook || undefined,
                trial_id: currentTrial || undefined,
                save_result: !!currentNotebook,
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(safeStr(result.error, 'Analysis failed'));
        }

        // Store for Excel export
        lastAnalysisResult = {
            analysis_type: backendType,
            analysis_id: backendId,
            config: config,
            summary: result.summary || '',
            statistics: result.statistics || {},
            plots: result.plots || [],
        };

        if (result.summary) {
            const session = document.getElementById('session-output');
            const wrapper = document.createElement('div');
            wrapper.style.position = 'relative';

            const div = document.createElement('div');
            div.className = 'result';

            // Check if summary has color tags
            if (result.summary.includes('<<COLOR:')) {
                const formatted = formatColoredOutput(result.summary);
                div.style.fontFamily = "'JetBrains Mono', 'Consolas', monospace";
                div.style.whiteSpace = 'pre';
                div.style.lineHeight = '1.4';
                div.innerHTML = formatted;
            } else if (result.summary.includes('<') || result.summary.includes('&')) {
                // Summary contains HTML markup (e.g., <strong>, &mdash;)
                div.innerHTML = result.summary;
            } else {
                div.textContent = result.summary;
            }

            // Copy button
            const copyBtn = document.createElement('button');
            copyBtn.className = 'aw-copy-btn';
            copyBtn.textContent = 'Copy';
            copyBtn.title = 'Copy to clipboard';
            copyBtn.onclick = () => {
                const plain = result.summary.replace(/<<COLOR:\w+>>/g, '').replace(/<<\/COLOR>>/g, '');
                navigator.clipboard.writeText(plain).then(() => {
                    copyBtn.textContent = 'Copied';
                    setTimeout(() => copyBtn.textContent = 'Copy', 1500);
                });
            };

            // Excel export button
            const xlsBtn = document.createElement('button');
            xlsBtn.className = 'aw-copy-btn';
            xlsBtn.style.right = '52px';
            xlsBtn.textContent = 'Excel';
            xlsBtn.title = 'Download as .xlsx';
            xlsBtn.onclick = () => {
                if (!lastAnalysisResult) return;
                xlsBtn.textContent = '...';
                fetch('/api/analysis/export/xlsx/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                    body: JSON.stringify(lastAnalysisResult),
                }).then(r => {
                    if (!r.ok) throw new Error('Export failed');
                    return r.blob();
                }).then(blob => {
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `svend_${lastAnalysisResult.analysis_id || 'analysis'}.xlsx`;
                    a.click();
                    URL.revokeObjectURL(url);
                    xlsBtn.textContent = 'Excel';
                }).catch(() => { xlsBtn.textContent = 'Failed'; setTimeout(() => { xlsBtn.textContent = 'Excel'; }, 1500); });
            };

            wrapper.appendChild(xlsBtn);
            wrapper.appendChild(copyBtn);
            wrapper.appendChild(div);
            session.appendChild(wrapper);
            session.scrollTop = session.scrollHeight;
        }

        if (result.plots && result.plots.length > 0) {
            displayGraphs(result.plots);
        }

        // Show save model button if model can be saved
        if (result.can_save && result.model_key) {
            showSaveModelButton(result.model_key, item?.name || currentAnalysis.id);
        }

        // Show Synara integration for Bayesian weights (only if project selected)
        if (result.synara_weights && currentProject) {
            const sw = result.synara_weights;
            showSynaraIntegration(sw.coefficients, sw.coefficients.map(c => c.feature), sw.target);
        }

        // Show statistics evidence UI (only if project selected)
        if (currentProject) {
            const stats = extractStatistics(result, currentAnalysis.id);
            // Also add explicit statistics from result
            if (result.statistics) {
                for (const [name, value] of Object.entries(result.statistics)) {
                    if (typeof value === 'number' && !stats.find(s => s.name === name)) {
                        stats.push({ name, value, variable: '' });
                    } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                        // Flatten nested dicts (e.g. group_means, group_stds)
                        for (const [subKey, subVal] of Object.entries(value)) {
                            if (typeof subVal === 'number') {
                                const flatName = `${name}.${subKey}`;
                                if (!stats.find(s => s.name === flatName)) {
                                    stats.push({ name: flatName, value: subVal, variable: '' });
                                }
                            }
                        }
                    }
                }
            }
            if (stats.length > 0) {
                showStatisticsEvidence(stats);
            }
        }

        // DSW enrichment blocks (INIT-009 / E9-004)
        renderDSWBlocks(result);

        addToHistory(currentAnalysis, item?.name, result.summary || '', result.plots || []);

        // Next steps suggestions
        showNextSteps(currentAnalysis.id, currentAnalysis.type);

        // Bayesian comparison for Gage R&R
        if (currentAnalysis.id === 'gage_rr' && config.compare_bayesian) {
            await runBayesianGRRComparison(config, result);
        }

        // Save to Measurement System button for GRR results
        if (['gage_rr', 'bayes_msa'].includes(currentAnalysis.id) && result.statistics) {
            showSaveMeasurementSystem(result);
        }

    } catch (err) {
        appendSession('error', `Error: ${safeStr(err, 'Unknown error')}`);
    }
}

// ── Bayesian GRR Comparison ──────────────────────────────────────────────

async function runBayesianGRRComparison(config, anovaResult) {
    appendSession('separator');
    appendSession('cmd', 'Bayesian Gage R&R (Comparison)');

    try {
        const bayesConfig = {
            measurement: config.measurement,
            part: config.part,
            operator: config.operator,
        };

        const response = await svendFetch('/api/analysis/run/', {
            method: 'POST',
            body: JSON.stringify({
                type: 'bayes_msa',
                analysis: 'bayes_msa',
                config: bayesConfig,
                data_id: currentData?.id,
            }),
        });

        if (response.summary) {
            const session = document.getElementById('session-output');
            const div = document.createElement('div');
            div.className = 'result';
            if (response.summary.includes('<') || response.summary.includes('&')) {
                div.innerHTML = response.summary;
            } else {
                div.textContent = response.summary;
            }
            session.appendChild(div);
            session.scrollTop = session.scrollHeight;
        }

        if (response.plots && response.plots.length > 0) {
            displayGraphs(response.plots);
        }

        // Comparison summary
        if (anovaResult.statistics && response.statistics) {
            const anova_grr = anovaResult.statistics.grr_percent || anovaResult.statistics.pct_grr;
            const bayes_grr = response.statistics.pct_grr_mean;
            const p_lt_10 = response.statistics.p_grr_lt_10;
            const p_lt_30 = response.statistics.p_grr_lt_30;

            if (anova_grr != null && bayes_grr != null) {
                let html = `<div style="background:rgba(74,159,110,0.08);border:1px solid rgba(74,159,110,0.2);border-radius:6px;padding:12px;margin-top:8px;font-size:12px;">`;
                html += `<strong style="color:var(--aw-accent);">Method Comparison</strong><br>`;
                html += `ANOVA %GRR: <strong>${Number(anova_grr).toFixed(1)}%</strong> (point estimate)<br>`;
                html += `Bayesian %GRR: <strong>${Number(bayes_grr).toFixed(1)}%</strong>`;
                if (response.statistics.pct_grr_ci) {
                    const ci = response.statistics.pct_grr_ci;
                    html += ` [${Number(ci[0]).toFixed(1)}%, ${Number(ci[1]).toFixed(1)}%]`;
                }
                html += `<br>`;
                if (p_lt_10 != null) html += `P(%GRR &lt; 10%): <strong>${(p_lt_10 * 100).toFixed(0)}%</strong> &mdash; `;
                if (p_lt_30 != null) html += `P(%GRR &lt; 30%): <strong>${(p_lt_30 * 100).toFixed(0)}%</strong>`;
                html += `</div>`;

                const session = document.getElementById('session-output');
                const wrap = document.createElement('div');
                wrap.innerHTML = html;
                session.appendChild(wrap);
            }
        }
    } catch (err) {
        appendSession('error', `Bayesian comparison failed: ${safeStr(err, 'Unknown error')}`);
    }
}

// ── Save to Measurement System ──────────────────────────────────────────


// ── Config value collection (from original executeAnalysis) ──
function collectConfigValues() {
    const config = {};
    document.querySelectorAll('#config-pane input[id], #config-pane select').forEach(el => {
        const key = el.id.replace('cfg-', '');
        if (el.multiple) {
            config[key] = Array.from(el.selectedOptions).map(o => o.value);
        } else if (el.type === 'checkbox' && el.id) {
            config[key] = el.checked;
        } else if (el.type !== 'checkbox') {
            config[key] = el.value;
        }
    });
    // Collect checkboxes by name (predictors, features, etc.)
    document.querySelectorAll('#config-pane input[type="checkbox"]:checked').forEach(cb => {
        const name = cb.name;
        if (!name) return;
        if (!config[name]) config[name] = [];
        config[name].push(cb.value);
    });
    // Plan rows for multiple_plan_comparison
    const planRows = document.querySelectorAll('#plan-rows .aw-form-row');
    if (planRows.length) {
        config.plans = [];
        planRows.forEach(row => {
            const name = row.querySelector('.plan-name')?.value || 'Plan';
            const n = parseInt(row.querySelector('.plan-n')?.value) || 50;
            const c = parseInt(row.querySelector('.plan-c')?.value) || 2;
            config.plans.push({ name, n, sample_size: n, c, accept_number: c, type: 'single' });
        });
    }
    return config;
}

// ── Smart column auto-select ──
function autoSelectColumns(columns) {
    const hints = {
        'cfg-response': /response|y|output|result|measure|value|yield|strength|weight/i,
        'cfg-var': /response|y|value|measure|result|data|output|yield/i,
        'cfg-var1': /x|before|method.?1|sample.?1|reference/i,
        'cfg-var2': /y|after|method.?2|sample.?2/i,
        'cfg-factor': /factor|group|treatment|category|type|machine|operator|batch/i,
        'cfg-group': /group|factor|category|treatment|type|batch/i,
        'cfg-measurement': /measurement|value|reading|data|response|result/i,
        'cfg-part': /part|sample|item|unit|piece/i,
        'cfg-operator': /operator|appraiser|inspector|rater|judge/i,
        'cfg-subject': /subject|id|patient|participant|person/i,
        'cfg-x': /x|factor.?1|temp|pressure|time|speed/i,
        'cfg-y': /y|factor.?2|flow|force|voltage/i,
        'cfg-z': /z|response|output|result|yield/i,
        'cfg-defectives': /defect|fail|reject|nonconform/i,
        'cfg-defects': /defect|count|error|flaw/i,
        'cfg-sample_size': /sample|size|n|count|units/i,
        'cfg-units': /units|area|length|sample/i,
        'cfg-time_column': /time|date|period|sequence|order/i,
    };
    document.querySelectorAll('#config-pane select[id]').forEach(sel => {
        const pattern = hints[sel.id];
        if (!pattern) return;
        const match = Array.from(sel.options).find(o => pattern.test(o.value));
        if (match) sel.value = match.value;
    });
}
