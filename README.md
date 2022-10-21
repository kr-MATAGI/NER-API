# NER-API
Named Enttiy Recognition API

### Python 패키지 종합 설치

```
pip install -r requirements.txt
```

### Fast API 실행 방법

```
uvicorn main:app --reload --host=[IP] --port=[PORT]
```

### POST

```
http://[IP]/ner
```

Request
```json
  {
     "date": "yyyy-mm-dd_hh:mm:ss.msec",
     "sentences": [
      {"id": "str or integer", "text": "str"}
     ]
  }
```
Response
```json
  {
    "date": "yyyy-mm-dd_hh:mm:ss.msec",
    "results": [{
      "id": "str", 
      "text": "str",
      "ne": [
        {"word": "str", "label": "str", "begin": "integer", "end": "integer"}
      ]
    }]
  }
```

### Test Page

```
http://127.0.0.1:8000/docs
```
