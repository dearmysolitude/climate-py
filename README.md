# [토이 프로젝트] 기후 데이터 시계열 분석
환경 데이터를 활용하여 시계열 분석을 진행하고 그 결과를 바탕으로 앞으로의 추세에 대한 예측을 해본다.

## 데이터 출처
  - CO2 농도 측정 데이터(1958년 ~ 2023년, [Global Monitoring Laboratory](https://gml.noaa.gov/ccgg/trends/))
  - 서울시 일일 기온 측정 데이터(1907년 ~ 2023년, [기상청 기상자료개방포털](https://data.kma.go.kr/stcs/grnd/grndTaList.do))
  - 전세계 해수면 위치 변화(1985년 ~ 2023년, [뉴스타파 DATA 포털/Correctiv](https://data.newstapa.org/datasets/%EC%A0%84%EC%84%B8%EA%B3%84-%ED%95%B4%EC%88%98%EB%A9%B4-%EB%86%92%EC%9D%B4-%EB%B3%80%ED%99%94-%EB%8D%B0%EC%9D%B4%ED%84%B0))

## 개발 환경
`./environment.yml` 파일 참고

## 요구 사항
- 하나의 페이지에서 모든 요청을 처리한다.
- 원하는 데이터에 대한 데이터 설정과 분석 버튼을 둔다.
- 데이터 업로드/처리는 버튼이 눌리면 진행된다.