import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/header';
import './App.css';
import Home from './components/Home';
import EnterData from './components/EnterData';
import LogViewer from './components/LogViewer';

const App = () => {
  return (
    <Router>
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/data-girisi" element={<EnterData />} />
        <Route path="/logs" element={<LogViewer />} />
        {/*
        <Route path="/dashboard" element={<Dashboard />} />

        <Route path="/kullanici" element={<Home />} />

        <Route path="/kullanici" element={<Home />} />

        <Route path="/kullanici" element={<Home />} />
        */}
      </Routes>
    </Router>
  );
};

export default App;
