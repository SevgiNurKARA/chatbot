import React from "react";
import { useNavigate } from "react-router-dom";
import "./header.css"
import koskiLogo from "../images/Koski-logo.png"
import BenimŞehrim from "../images/BenimŞehrim.png"
import KonyaBüyükşehirBelediyesi from "../images/KonyaBüyükşehirBelediyesi.png"

const Header = () => {
  const navigate = useNavigate();

  const navigateTo = (path) => {
    navigate(path);
  };

  return (
    <header className="header">
      <div className="Companylogo">
        <img className='logo-images' src={koskiLogo} alt="logo" />
        <div className="companyName">KONYA SU VE KANALİZASYON
        İDARESİ GENEL MÜDÜRLÜĞÜ</div>
      </div>
      <div className="selection-parts">
        <button className="buttonHeader" onClick={() => navigateTo('/')}> Ana sayfa</button>
        <button className="buttonHeader" onClick={() => navigateTo('/data-girisi')}> Data girişi</button>
        <button className="buttonHeader" onClick={() => navigateTo('/dashboard')}> Dashboard</button>
        <button className="buttonHeader" onClick={() => navigateTo('/kullanici')}> Kullanıcı</button>
        <button className="buttonHeader" onClick={() => navigateTo('/kullanici')}> Kullanıcı</button>
      </div>
      <div className="logo-parts">
      
      <img className='logoPartsLogos' src={BenimŞehrim} alt="logo" />
      <img className='logoPartsLogos' src={KonyaBüyükşehirBelediyesi} alt="logo" />
      <img className='logoPartsLogos' src={koskiLogo} alt="logo" />
      </div>
    </header>
  );
};

export default Header;
