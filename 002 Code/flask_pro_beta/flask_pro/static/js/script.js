// static/js/script.js

// ==== DOMContentLoaded 이벤트 리스너: 페이지 로드 시 필요한 함수 호출 ====
document.addEventListener('DOMContentLoaded', () => {
    const path = window.location.pathname;
    if (path === '/' || path === '/index.html') {
        loadExpenses();
    } else if (path === '/add_expense' || path === '/add_expense.html') {
        // 'add_expense.html' 페이지에서만 category-select를 로드하고 이벤트 리스너를 설정
        setupCategorySelect();
        setupFormListeners();
    } else if (path === '/analysis' || path === '/analysis.html') {
        loadAnalysis();
    }
});


// ==== 지출 목록 로드 함수 (index.html 용) ====
async function loadExpenses() {
    const expenseListDiv = document.getElementById('expense-list');
    if (!expenseListDiv) return; // add_expense.html이나 analysis.html에서는 이 요소가 없음

    try {
        // Flask API 엔드포인트로 변경
        const response = await fetch('/api/get_expenses');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const expenses = await response.json();

        expenseListDiv.innerHTML = ''; // 기존 내용 삭제

        if (expenses.length === 0) {
            expenseListDiv.innerHTML = '<p>아직 지출 내역이 없습니다. 새로운 지출을 추가해보세요!</p>';
            return;
        }

        expenses.forEach(expense => {
            const expenseItem = document.createElement('div');
            expenseItem.className = 'expense-item';
            expenseItem.innerHTML = `
                <p><strong>날짜:</strong> ${expense.날짜}</p>
                <p><strong>카테고리:</strong> ${expense.카테고리}</p>
                <p><strong>내용:</strong> ${expense.음식명}</p>
                <p><strong>금액:</strong> ${expense.금액.toLocaleString()}원</p>
                <p><strong>만족도:</strong> ${'⭐'.repeat(expense.만족도)}</p>
                ${expense.추천 ? `<p class="recommendation-text"><strong>추천:</strong> ${expense.추천}</p>` : ''}
            `;
            expenseListDiv.appendChild(expenseItem);
        });
    } catch (error) {
        console.error('지출 내역을 불러오는 중 오류 발생:', error);
        expenseListDiv.innerHTML = '<p>지출 내역을 불러오는 데 실패했습니다. 잠시 후 다시 시도해주세요.</p>';
    }
}


// ==== 지출 추가 폼 관련 함수 (add_expense.html 용) ====

// 카테고리 선택지 설정
function setupCategorySelect() {
    const categorySelect = document.getElementById('category');
    if (!categorySelect) return;

    const categories = ['카페', '식비', '편의점', '마트', '술/유흥', '간식', '배달음식', '교통비', '기타'];
    categories.forEach(category => {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category;
        categorySelect.appendChild(option);
    });
}

// 폼 제출 이벤트 리스너 설정
function setupFormListeners() {
    const addExpenseForm = document.getElementById('add-expense-form');
    if (addExpenseForm) {
        addExpenseForm.addEventListener('submit', async (event) => {
            event.preventDefault(); // 폼 기본 제출 동작 방지

            const date = document.getElementById('expense-date').value;
            const category = document.getElementById('category').value;
            const itemName = document.getElementById('item-name').value;
            const amount = document.getElementById('amount').value;
            const satisfaction = document.getElementById('satisfaction').value;

            // 필수 입력 필드 유효성 검사
            if (!date || !category || !itemName || !amount || !satisfaction) {
                alert('모든 필드를 입력해주세요!');
                return;
            }
            if (isNaN(amount) || parseInt(amount) <= 0) {
                alert('금액은 유효한 숫자로 입력해주세요.');
                return;
            }
            if (isNaN(satisfaction) || parseInt(satisfaction) < 1 || parseInt(satisfaction) > 5) {
                alert('만족도는 1에서 5 사이의 숫자로 입력해주세요.');
                return;
            }

            try {
                // Flask API 엔드포인트로 변경
                const response = await fetch('/api/add_expense', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    // JSON.stringify로 데이터를 직렬화하여 전송
                    body: JSON.stringify({
                        date: date,
                        category: category,
                        itemName: itemName,
                        amount: parseInt(amount), // 숫자로 변환
                        satisfaction: parseInt(satisfaction) // 숫자로 변환
                    }),
                });

                if (!response.ok) {
                    // 서버에서 에러 메시지를 JSON으로 보낼 경우를 대비
                    const errorData = await response.json().catch(() => response.text());
                    throw new Error(`HTTP error! status: ${response.status}, message: ${JSON.stringify(errorData)}`);
                }

                const result = await response.json();
                alert(result.message + '\n\n' + (result.recommendation || ''));
                window.location.href = '/'; // 메인 페이지로 리디렉션
            } catch (error) {
                console.error('지출 추가 중 오류 발생:', error);
                alert('지출 추가에 실패했습니다. 오류: ' + error.message);
            }
        });
    }
}


// ==== 지출 분석 로드 함수 (analysis.html 용) ====
async function loadAnalysis() {
    const analysisResultsDiv = document.getElementById('analysis-results');
    if (!analysisResultsDiv) return;

    try {
        // Flask API 엔드포인트로 변경
        const response = await fetch('/api/analysis');
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }
        const data = await response.json();

        // 데이터가 없는 경우 처리 (서버에서 404를 보낼 수도 있지만, 여기서도 한번 더 체크)
        if (data.error) {
            analysisResultsDiv.innerHTML = `<p>${data.error}</p>`;
            return;
        }

        analysisResultsDiv.innerHTML = `
            <h2>지출 요약</h2>
            <p><strong>총 소비액:</strong> ${data.total_expense.toLocaleString()}원</p>
            <p><strong>평균 일일 소비액:</strong> ${data.avg_daily_expense.toLocaleString()}원 (일일 예산: ${data.daily_budget.toLocaleString()}원)</p>
        `;

        if (data.overspent_days && data.overspent_days.length > 0) {
            let overspentHtml = '<h3>예산 초과 날짜:</h3><ul>';
            data.overspent_days.forEach(day => {
                overspentHtml += `<li>${day.date}: ${day.amount.toLocaleString()}원 (초과: ${day.over_amount.toLocaleString()}원)</li>`;
            });
            overspentHtml += '</ul>';
            analysisResultsDiv.innerHTML += overspentHtml;
        } else {
            analysisResultsDiv.innerHTML += '<p>모든 날짜에서 예산 내에서 소비했습니다!</p>';
        }

        if (data.category_total && data.category_total.length > 0) {
            let categoryHtml = '<h3>카테고리별 지출:</h3><ul>';
            data.category_total.forEach(cat => {
                categoryHtml += `<li>${cat.category}: ${cat.amount.toLocaleString()}원 (${cat.percentage}%)</li>`;
            });
            categoryHtml += '</ul>';
            analysisResultsDiv.innerHTML += categoryHtml;
        }

        if (data.weekday_avg && data.weekday_avg.length > 0) {
            let weekdayHtml = '<h3>요일별 평균 지출:</h3><ul>';
            data.weekday_avg.forEach(day => {
                weekdayHtml += `<li>${day.day}: ${day.avg_amount.toLocaleString()}원</li>`;
            });
            weekdayHtml += '</ul>';
            analysisResultsDiv.innerHTML += weekdayHtml;
        }

    } catch (error) {
        console.error('지출 분석 데이터를 불러오는 중 오류 발생:', error);
        analysisResultsDiv.innerHTML = '<p>지출 분석 데이터를 불러오는 데 실패했습니다. 잠시 후 다시 시도해주세요.</p>';
    }
}