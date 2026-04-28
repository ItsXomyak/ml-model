const form = document.getElementById('apt-form');
const resultEl = document.getElementById('result');
const errorEl = document.getElementById('error');
const submitBtn = document.getElementById('submit-btn');
const modelInfo = document.getElementById('model-info');

// Подгружаем инфу о модели при старте
fetch('/api/health')
    .then(r => r.json())
    .then(data => {
        if (data.status === 'ok') {
            const r2 = data.test_metrics?.r2;
            const mse = data.test_metrics?.mse;
            modelInfo.textContent =
                `модель: ${data.model_used}  ·  R² test: ${r2?.toFixed(4)}  ·  MSE: ${mse?.toFixed(2)}  ·  K=${data.optimal_k}`;
        } else {
            modelInfo.textContent = `⚠ ${data.detail || 'Модели не загружены'}`;
        }
    })
    .catch(() => {
        modelInfo.textContent = '⚠ API недоступен';
    });

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    resultEl.classList.add('hidden');
    errorEl.classList.add('hidden');

    const formData = new FormData(form);
    const payload = {};
    for (const [key, value] of formData.entries()) {
        // Числовые поля парсим в числа
        if (['area', 'rooms', 'floor', 'total_floors', 'year_built', 'dist_to_center'].includes(key)) {
            payload[key] = Number(value);
        } else {
            payload[key] = value;
        }
    }

    submitBtn.disabled = true;
    submitBtn.textContent = 'Считаю...';

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || `HTTP ${res.status}`);
        }

        renderResult(data);
    } catch (err) {
        errorEl.textContent = `Ошибка: ${err.message}`;
        errorEl.classList.remove('hidden');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Оценить →';
    }
});

function renderResult(data) {
    resultEl.innerHTML = `
        <div class="price">${data.price_mln.toFixed(2)}</div>
        <div class="currency">млн ₸</div>
        <div class="class-badge">${data.class_name}</div>
        <div class="meta">
            <span>Цена за м²: <strong>${(data.price_per_m2 * 1000).toFixed(0)} тыс ₸</strong></span>
            <span>Кластер: <strong>#${data.cluster_id}</strong></span>
            <span>Модель: <strong>${data.model_used}</strong></span>
        </div>
    `;
    resultEl.classList.remove('hidden');
    resultEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
