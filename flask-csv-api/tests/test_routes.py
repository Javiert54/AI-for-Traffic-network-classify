import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_upload_csv(client):
    data = {
        'file': (open('tests/test_file.csv', 'rb'), 'test_file.csv')
    }
    response = client.post('/upload', data=data)
    assert response.status_code == 200
    assert b'CSV file processed successfully' in response.data

def test_upload_invalid_file(client):
    data = {
        'file': (open('tests/invalid_file.txt', 'rb'), 'invalid_file.txt')
    }
    response = client.post('/upload', data=data)
    assert response.status_code == 400
    assert b'Invalid file type' in response.data

def test_upload_no_file(client):
    response = client.post('/upload')
    assert response.status_code == 400
    assert b'No file part' in response.data