# Upbit Volatility Breakout Trading Bot (업비트 변동성 돌파 트레이딩 봇)

**변동성 돌파(Volatility Breakout, VB)** 전략과 **변동성 타겟팅(Volatility Targeting)** 자금 관리 기법을 활용한 고성능 업비트 자동매매 봇입니다. 실시간 웹 대시보드를 통해 포트폴리오와 봇의 상태를 모니터링할 수 있습니다.

## 🚀 주요 기능

*   **알고리즘 트레이딩**:
    *   **전략**: 래리 윌리엄스(Larry Williams)의 변동성 돌파 전략 사용.
    *   **동적 K (Dynamic K)**: 최근 시장의 노이즈 비율에 따라 돌파 계수(K)를 자동으로 조정.
    *   **변동성 타겟팅 (Volatility Targeting)**: 자산의 변동성(ATR)에 따라 포지션 규모를 동적으로 조절하여 리스크 관리.
    *   **트레일링 스탑 (Trailing Stop)**: 가격 상승 시 익절 라인을 따라 올리며 수익 보존.
    *   **급등/급락 보호**: 펌프 앤 덤프(Pump & Dump) 감지 필터 적용.
*   **웹 대시보드**:
    *   보유 자산, 미실현 손익(PnL), 미체결 주문 실시간 조회.
    *   **Market Watch**: 감시 중인 코인의 상태(목표가, 이동평균, 대기/매수신호 등) 실시간 확인.
    *   30초 자동 새로고침 기능.
*   **배포 및 운영**:
    *   Docker 지원으로 24/7 안정적인 서버 운영 가능.
    *   자동 재시작 메커니즘 포함.
*   **백테스팅 (Backtesting)**:
    *   내장된 시뮬레이션 도구(`backtest.py`)를 통해 최근 30일 데이터 기반 전략 검증 가능.

## 🛠 필수 요구 사항

*   Python 3.9 이상
*   [Upbit](https://upbit.com/) 계정 및 API Key (Access/Secret Key)
*   Docker (선택 사항, 서버 배포 시 권장)

## ⚙️ 설치 및 설정

1.  **리포지토리 클론 (Clone)**
    ```bash
    git clone https://github.com/your-username/upbit-trade.git
    cd upbit-trade
    ```

2.  **환경 변수 설정**
    프로젝트 루트 경로에 `.env` 파일을 생성하고 업비트 API 키를 입력하세요:
    ```bash
    # .env 파일 생성
    access_key=YOUR_UPBIT_ACCESS_KEY
    secret_key=YOUR_UPBIT_SECRET_KEY
    PAPER_MODE=True  # 실전 매매 시 False로 변경
    ```

## 🏃‍♂️ 실행 방법

### 방법 1: Docker (권장)
서버에서 24시간 중단 없이 실행하기 가장 좋은 방법입니다.

1.  **이미지 빌드**
    ```bash
    docker build -t upbit-trade .
    ```

2.  **컨테이너 실행**
    ```bash
    docker run -d --name upbit-trade \
      --env-file .env \
      -v $(pwd)/trades.db:/app/trades.db \
      -v $(pwd)/trade.log:/app/trade.log \
      -v $(pwd)/bot_status.json:/app/bot_status.json \
      -p 5000:5000 \
      upbit-trade
    ```

### 방법 2: 로컬 실행
1.  **라이브러리 설치**
    ```bash
    pip install -r requirements.txt
    ```

2.  **봇 시작**
    ```bash
    ./start.sh
    ```
    *   이 스크립트는 트레이딩 봇을 백그라운드에서 실행하고, 웹 대시보드를 포그라운드에서 실행합니다.
    *   대시보드 접속: `http://localhost:5000`

## 📊 대시보드

실행 후 브라우저에서 `http://localhost:5000` (또는 서버 IP)에 접속하면 다음 정보를 확인할 수 있습니다:
*   **Asset Summary**: 총 추정 자산 및 주문 가능 원화.
*   **Market Watch**: 감시 종목의 실시간 지표 (목표가, 이격도, 매수 신호 여부 등).
*   **Holdings**: 현재 보유 중인 코인과 수익률.
*   **Trade History**: 최근 매수/매도 체결 내역.

## 🧪 백테스팅 (Backtesting)

과거 데이터를 바탕으로 전략을 검증하려면 아래 명령어를 실행하세요:
```bash
python backtest.py
```
*   최근 30일간의 1분봉 데이터를 수집하여 시뮬레이션을 수행하고, 예상 수익률을 출력합니다.

## 📁 프로젝트 구조

*   `trade.py`: 봇의 핵심 매매 로직.
*   `app.py`: 웹 대시보드 구동을 위한 Flask 서버.
*   `backtest.py`: 백테스팅 시뮬레이션 스크립트.
*   `start.sh`: 프로세스 실행 관리 스크립트.
*   `templates/`: 대시보드용 HTML 템플릿.

## ⚠️ 면책 조항 (Disclaimer)

이 소프트웨어는 교육 및 참고 목적으로 제작되었습니다. 투자의 책임은 전적으로 사용자 본인에게 있습니다. 본 소프트웨어를 사용하여 발생한 금전적 손실에 대해 개발자는 어떠한 책임도 지지 않습니다. 신중하게 테스트하고 사용하시기 바랍니다.
