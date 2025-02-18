import React, { useState, useEffect } from 'react';
import "./EnterData.css"

const EnterData = () => {
  // State tanımlamaları
  const [intent, setIntent] = useState('');
  const [examples, setExamples] = useState('');
  const [responses, setResponses] = useState('');
  const [chatbotData, setChatbotData] = useState({ intents: [] });
  const [message, setMessage] = useState('');

  // Mevcut chatbot verilerini yükle
  useEffect(() => {
    fetchChatbotData();
  }, []);

  // Chatbot verilerini getir
  const fetchChatbotData = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/chatbot/data');
      const data = await response.json();
      setChatbotData(data);
    } catch (error) {
      console.error('Veri getirme hatası:', error);
      setMessage('Veriler yüklenirken bir hata oluştu');
    }
  };

  // Yeni intent ekle
  const handleAddIntent = async () => {
    if (!intent || !examples || !responses) {
      setMessage('Lütfen tüm alanları doldurun');
      return;
    }

    try {
      // Yeni intent'i hazırla
      const newIntent = {
        intent: intent.trim(),
        examples: examples.split('\n').map(ex => ex.trim()).filter(ex => ex), // Boş satırları filtrele
        responses: responses.split('\n').map(res => res.trim()).filter(res => res)
      };

      // Mevcut intent'lere ekle
      const updatedData = {
        intents: [...chatbotData.intents, newIntent]
      };

      // API'ye gönder
      const response = await fetch('http://localhost:5000/api/chatbot/data', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedData)
      });

      if (response.ok) {
        // Başarılı ise state'i güncelle
        setChatbotData(updatedData);
        setMessage('Veri başarıyla eklendi');
        
        // Form alanlarını temizle
        setIntent('');
        setExamples('');
        setResponses('');
      } else {
        const error = await response.json();
        setMessage(`Hata: ${error.error}`);
      }
    } catch (error) {
      console.error('Veri ekleme hatası:', error);
      setMessage('Veri eklenirken bir hata oluştu');
    }
  };

  return (
    <div className="EnterData-container">
      <div className='EnterData-inputs-part'>
        <div className='EnterData-İnputs-part-colums' style={{height: '400px'}}>
          <div className='EnterData-inputs-part-parts'>
            <p className='EnterData-input-part-titels'>Başlık:</p>
            <textarea 
              placeholder="Intent başlığını girin" 
              className='EnterData-input-box-title'
              value={intent}
              onChange={(e) => setIntent(e.target.value)}
            />
          </div>
          <button 
            className='EnterData-input-buttons'
            onClick={handleAddIntent}
          >
            EKLE
          </button>
          {message && (
            <div className={`message ${message.includes('Hata') ? 'error' : 'success'}`}>
              {message}
            </div>
          )}
        </div>
        <div className='EnterData-İnputs-part-colums'>
          <div className='EnterData-inputs-part-parts'>
            <p className='EnterData-input-part-titels'>Sorular:</p>
            <textarea 
              placeholder="Her satıra bir soru yazın" 
              className='EnterData-input-box-questions'
              value={examples}
              onChange={(e) => setExamples(e.target.value)}
            />
          </div>
          <div className='EnterData-inputs-part-parts'>
            <p className='EnterData-input-part-titels'>Cevaplar:</p>
            <textarea 
              placeholder="Her satıra bir cevap yazın" 
              className='EnterData-input-box-answers'
              value={responses}
              onChange={(e) => setResponses(e.target.value)}
            />
          </div>
        </div>
      </div>

      {/* Mevcut verileri göster */}
      <div className="EnterData-current-data">
        <h3>Son Eklenen Başlıklar</h3>
        <div className="intent-list">
          {chatbotData.intents
            .slice(-3) // Son 3 intent'i al
            .reverse() // En son eklenen en üstte görünsün
            .map((item, index) => (
              <div key={index} className="intent-item">
                <h4>{item.intent}</h4>
                <div className="intent-details">
                  <div>
                    <strong>Sorular:</strong>
                    <ul>
                      {item.examples.map((ex, i) => (
                        <li key={i}>{ex}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <strong>Cevaplar:</strong>
                    <ul>
                      {item.responses.map((res, i) => (
                        <li key={i}>{res}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
          ))}
        </div>
        {chatbotData.intents.length > 3 && (
          <div className="more-intents-info">
            <p>Toplam {chatbotData.intents.length} başlık bulunmaktadır.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default EnterData;