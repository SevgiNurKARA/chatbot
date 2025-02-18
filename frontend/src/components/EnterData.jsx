import React from 'react';
import "./EnterData.css"

const EnterData = () => {
  return (
    <div className="EnterData-container">
      <div className='EnterData-inputs-part'>
        <div className='EnterData-İnputs-part-colums' style={{height: '400px'}}>
          <div className='EnterData-inputs-part-parts'>
          <p className='EnterData-input-part-titels'>Başlık:</p>
          <textarea placeholder="Bir cümle yazın" className='EnterData-input-box-title'></textarea>
          
        </div>
        <button className='EnterData-input-buttons'>EKLE</button>
        </div>
        <div className='EnterData-İnputs-part-colums'>
          <div className='EnterData-inputs-part-parts'>
           <p className='EnterData-input-part-titels'>Sorular:</p>
          <textarea placeholder="Bir cümle yazın" className='EnterData-input-box-questions'></textarea>
        </div>
        <div className='EnterData-inputs-part-parts'>
          <p className='EnterData-input-part-titels'>cevaplar:</p>
          <textarea placeholder="Bir cümle yazın" className='EnterData-input-box-answers'></textarea>
        </div>
        </div>
        
        
        </div>
    </div>
  );
};

export default EnterData;