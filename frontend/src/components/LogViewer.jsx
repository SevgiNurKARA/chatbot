import React, { useState, useEffect } from 'react';
import './LogViewer.css';

const LogViewer = () => {
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchLogs();
        // Her 30 saniyede bir logları güncelle
        const interval = setInterval(fetchLogs, 30000);
        return () => clearInterval(interval);
    }, []);

    const fetchLogs = async () => {
        try {
            const response = await fetch('http://localhost:5000/api/logs');
            if (!response.ok) {
                throw new Error('Loglar alınamadı');
            }
            const data = await response.json();
            setLogs(data);
            setError(null);
        } catch (error) {
            console.error('Loglar yüklenirken hata oluştu:', error);
            setError('Loglar yüklenirken bir hata oluştu');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return <div className="loading">Loglar yükleniyor...</div>;
    }

    if (error) {
        return <div className="error">{error}</div>;
    }

    if (logs.length === 0) {
        return <div className="no-logs">Henüz log kaydı bulunmamaktadır.</div>;
    }

    return (
        <div className="log-viewer-container">
            <h2>Sistem Logları</h2>
            <div className="log-list">
                {logs.map((log, index) => (
                    <div key={index} className={`log-item ${log.level.toLowerCase()}`}>
                        <span className="log-timestamp">
                            {new Date(log.timestamp).toLocaleString()}
                        </span>
                        <span className={`log-level ${log.level}`}>
                            {log.level}
                        </span>
                        <span className="log-message">{log.message}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default LogViewer; 