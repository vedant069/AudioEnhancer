import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
});

export async function enhanceAudio(file: File): Promise<string> {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await api.post('/enhance-audio', formData, {
      responseType: 'blob',
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    if (!response.data) {
      throw new Error('No data received from server');
    }

    // Create a blob from the response data with the correct MIME type
    const audioBlob = new Blob([response.data], { type: 'audio/wav' });
    return URL.createObjectURL(audioBlob);
  } catch (error) {
    console.error('Error enhancing audio:', error);
    throw error;
  }
}