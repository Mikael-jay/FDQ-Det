const modelSelect = document.getElementById("modelSelect");
const imageInput = document.getElementById("imageInput");
const predictBtn = document.getElementById("predictBtn");
const statusEl = document.getElementById("status");
const originalImage = document.getElementById("originalImage");
const resultImage = document.getElementById("resultImage");
const confThresh = document.getElementById("confThresh");
const nmsIouThresh = document.getElementById("nmsIouThresh");
const customCheckpoint = document.getElementById("customCheckpoint");
const customConfig = document.getElementById("customConfig");
const selectedCount = document.getElementById("selectedCount");
const modelTemplateHint = document.getElementById("modelTemplateHint");
const resultList = document.getElementById("resultList");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const pageInfo = document.getElementById("pageInfo");

let loadedModels = [];
let objectUrls = [];
let currentItems = [];
let currentIndex = 0;
const splitter = document.getElementById("splitter");
const resultContainer = document.querySelector('.result');
const leftViewer = () => resultContainer && resultContainer.querySelectorAll('.viewer')[0];
const rightViewer = () => resultContainer && resultContainer.querySelectorAll('.viewer')[1];
let isDragging = false;
let dragStartX = 0;
let startLeftWidth = 50; // percent

function setStatus(message, asError = false) {
    statusEl.textContent = message;
    statusEl.style.color = asError ? "#b00020" : "#0f4c5c";
}

function cleanupObjectUrls() {
    objectUrls.forEach((url) => URL.revokeObjectURL(url));
    objectUrls = [];
}

function getModelTemplate(modelId) {
    const selected = loadedModels.find((m) => m.id === modelId);
    return selected?.default_params || null;
}

function applyModelTemplate(modelId) {
    const template = getModelTemplate(modelId);
    if (!template) {
        modelTemplateHint.textContent = "当前模型未配置参数模板";
        return;
    }
    if (typeof template.conf_thresh === "number") {
        confThresh.value = String(template.conf_thresh);
    }
    if (typeof template.nms_iou_thresh === "number") {
        nmsIouThresh.value = String(template.nms_iou_thresh);
    }
    modelTemplateHint.textContent = `已应用模板: conf=${confThresh.value}, nms=${nmsIouThresh.value}`;
}

function renderList() {
    resultList.innerHTML = "";
    currentItems.forEach((item, index) => {
        const li = document.createElement("li");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `resultItem ${item.ok ? "ok" : "fail"} ${index === currentIndex ? "active" : ""}`;
        const detCount = item.ok ? item.detections : "-";
        btn.textContent = `${index + 1}. ${item.filename} | det=${detCount} | ${item.ok ? "ok" : "error"}`;
        btn.addEventListener("click", () => {
            currentIndex = index;
            showCurrent();
        });
        li.appendChild(btn);
        resultList.appendChild(li);
    });
}

function showCurrent() {
    if (currentItems.length === 0) {
        pageInfo.textContent = "0 / 0";
        originalImage.removeAttribute("src");
        resultImage.removeAttribute("src");
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        return;
    }
    const item = currentItems[currentIndex];
    originalImage.src = item.originalUrl;
    if (item.ok) {
        resultImage.src = `data:image/jpeg;base64,${item.rendered}`;
        setStatus(
            `识别完成\n当前: ${currentIndex + 1}/${currentItems.length}\n文件: ${item.filename}\n目标数: ${item.detections}\n耗时: ${item.elapsed}`
        );
    } else {
        resultImage.removeAttribute("src");
        setStatus(`当前结果失败\n文件: ${item.filename}\n错误: ${item.error}`, true);
    }

    pageInfo.textContent = `${currentIndex + 1} / ${currentItems.length}`;
    prevBtn.disabled = currentIndex === 0;
    nextBtn.disabled = currentIndex >= currentItems.length - 1;
    renderList();
}

async function loadModels() {
    const resp = await fetch("/api/models");
    if (!resp.ok) {
        throw new Error(await resp.text());
    }

    const payload = await resp.json();
    loadedModels = payload.models || [];

    modelSelect.innerHTML = "";
    loadedModels.forEach((m) => {
        const option = document.createElement("option");
        option.value = m.id;
        option.textContent = `${m.name} (${m.dataset})`;
        modelSelect.appendChild(option);
    });

    if (loadedModels.length === 0) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "无可用模型，请修改 model_registry.json";
        modelSelect.appendChild(option);
        modelTemplateHint.textContent = "未加载到模型模板";
    } else {
        applyModelTemplate(modelSelect.value || loadedModels[0].id);
    }
}

modelSelect.addEventListener("change", () => {
    applyModelTemplate(modelSelect.value);
});

imageInput.addEventListener("change", () => {
    const files = Array.from(imageInput.files || []);
    cleanupObjectUrls();
    currentItems = [];
    currentIndex = 0;
    resultList.innerHTML = "";

    if (files.length === 0) {
        selectedCount.textContent = "当前未选择图片";
        showCurrent();
        return;
    }

    files.forEach((file) => {
        const url = URL.createObjectURL(file);
        objectUrls.push(url);
        currentItems.push({
            filename: file.name,
            originalUrl: url,
            ok: false,
            detections: "-",
            elapsed: "-",
            rendered: "",
            error: "尚未识别",
            file,
        });
    });

    selectedCount.textContent = `已选择 ${files.length} 张图片`;
    originalImage.src = currentItems[0].originalUrl;
    resultImage.removeAttribute("src");
    pageInfo.textContent = `1 / ${currentItems.length}`;
    prevBtn.disabled = true;
    nextBtn.disabled = currentItems.length <= 1;
    renderList();
    setStatus(`已加载 ${files.length} 张图片，点击“开始识别（批量）”执行推理`);
});

predictBtn.addEventListener("click", async () => {
    const files = Array.from(imageInput.files || []);
    if (files.length === 0) {
        setStatus("请先选择至少一张图片", true);
        return;
    }

    predictBtn.disabled = true;
    setStatus(`识别中，请稍候... (0/${files.length})`);

    try {
        const formData = new FormData();
        files.forEach((file) => formData.append("images", file));
        formData.append("model_id", modelSelect.value);
        formData.append("conf_thresh", confThresh.value || "0.55");
        formData.append("nms_iou_thresh", nmsIouThresh.value || "0.5");

        const checkpoint = customCheckpoint.value.trim();
        const config = customConfig.value.trim();
        if (checkpoint) {
            formData.append("custom_checkpoint", checkpoint);
        }
        if (config) {
            formData.append("custom_config", config);
        }

        const resp = await fetch("/api/predict_batch", {
            method: "POST",
            body: formData,
        });

        if (!resp.ok) {
            throw new Error(await resp.text());
        }

        const payload = await resp.json();
        currentItems = files.map((file, index) => {
            const item = payload.results?.[index] || { ok: false, error: "No result" };
            return {
                filename: file.name,
                originalUrl: currentItems[index]?.originalUrl || URL.createObjectURL(file),
                ok: Boolean(item.ok),
                detections: item.ok ? (item.result?.detections?.length ?? 0) : "-",
                elapsed: item.ok ? (item.result?.inference_time ?? "-") : "-",
                rendered: item.ok ? (item.rendered_image_base64 || "") : "",
                error: item.ok ? "" : (item.error || "Unknown error"),
            };
        });
        currentIndex = 0;
        showCurrent();
        setStatus(
            `批量识别完成\n模型: ${payload.model_id}\n数据集: ${payload.dataset}\n成功: ${payload.success}/${payload.total}\n失败: ${payload.failed}`,
            payload.failed > 0
        );
    } catch (error) {
        setStatus(`识别失败: ${error.message}`, true);
    } finally {
        predictBtn.disabled = false;
    }
});

prevBtn.addEventListener("click", () => {
    if (currentIndex > 0) {
        currentIndex -= 1;
        showCurrent();
    }
});

nextBtn.addEventListener("click", () => {
    if (currentIndex < currentItems.length - 1) {
        currentIndex += 1;
        showCurrent();
    }
});

(async function init() {
    try {
        await loadModels();
        showCurrent();
        setStatus("模型列表加载完成，请选择图片后开始识别");
    } catch (error) {
        setStatus(`初始化失败: ${error.message}`, true);
    }
})();

// Drag logic for splitter
function clamp(v, a, b) { return Math.min(Math.max(v, a), b); }

if (splitter && resultContainer) {
    splitter.addEventListener('pointerdown', (e) => {
        e.preventDefault();
        isDragging = true;
        dragStartX = e.clientX;
        const rect = resultContainer.getBoundingClientRect();
        const leftRect = leftViewer().getBoundingClientRect();
        startLeftWidth = (leftRect.width / rect.width) * 100;
        splitter.setPointerCapture(e.pointerId);
    });

    window.addEventListener('pointermove', (e) => {
        if (!isDragging) return;
        const rect = resultContainer.getBoundingClientRect();
        const deltaX = e.clientX - rect.left; // absolute position relative to container left
        const splitterPx = 12;
        // account for viewer padding/borders and ensure boxes stay within image area
        const minViewerPx = 220; // slightly larger to include chrome and padding

        // compute left column in pixels and clamp so it never becomes too small/large
        let leftPx = deltaX;
        leftPx = clamp(leftPx, minViewerPx, rect.width - minViewerPx - splitterPx);

        // set grid columns using a fixed pixel left column, fixed splitter, and auto right column
        resultContainer.style.gridTemplateColumns = `${leftPx}px ${splitterPx}px auto`;
    });

    window.addEventListener('pointerup', (e) => {
        if (!isDragging) return;
        isDragging = false;
        try { splitter.releasePointerCapture(e.pointerId); } catch (_) { }
    });

    // double-click resets to equal split
    splitter.addEventListener('dblclick', () => {
        resultContainer.style.gridTemplateColumns = '1fr 12px 1fr';
    });
}
