from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import sqlite3
from datetime import datetime
import os
import threading
import time
from functools import wraps

# Import emergency detection modules with graceful fallback
AudioStream = None
SpeechToText = None
TextProcessor = None
EmergencyClassifier = None
LLMReasoner = None
DecisionMaker = None

try:
    from audio_processing.audio_stream import AudioStream
except Exception as e:
    print(f"⚠️  AudioStream module not available: {type(e).__name__}")

try:
    from speech_to_text.stt import SpeechToText
except Exception as e:
    print(f"⚠️  SpeechToText module not available: {type(e).__name__}")

try:
    from nlp_processing.nlp import TextProcessor
except Exception as e:
    print(f"⚠️  TextProcessor module not available: {type(e).__name__}")

try:
    from classifier.classifier import EmergencyClassifier
except Exception as e:
    print(f"⚠️  EmergencyClassifier module not available: {type(e).__name__}")

try:
    from llm_reasoning.llm import LLMReasoner
except Exception as e:
    print(f"⚠️  LLMReasoner module not available: {type(e).__name__}")

try:
    from decision_logic.decision import DecisionMaker
except Exception as e:
    print(f"⚠️  DecisionMaker module not available: {type(e).__name__}")

app = Flask(__name__)
app.config['DATABASE'] = 'events.db'
app.config['SECRET_KEY'] = 'your_secret_key_here_change_in_production'

# Global variables for monitoring
monitoring_active = False
monitor_thread = None
emergency_events = []
system_status = {'monitoring': False, 'events_count': 0, 'last_event': None}

# Database initialization
def get_db_connection():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name TEXT NOT NULL,
            event_type TEXT NOT NULL,
            description TEXT,
            date_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            location TEXT,
            status TEXT DEFAULT 'Active',
            confidence REAL
        );
        
        CREATE TABLE IF NOT EXISTS emergency_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            audio_transcription TEXT,
            confidence REAL,
            decision TEXT,
            is_emergency INTEGER DEFAULT 0,
            action_taken TEXT
        );
        
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT,
            message TEXT,
            status TEXT
        );
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Emergency Detection System Class
class EmergencyDetectionSystem:
    def __init__(self):
        self.monitoring = False
        self.initialized = False
        
        # Check if all modules are available
        if all([AudioStream, SpeechToText, TextProcessor, EmergencyClassifier, LLMReasoner, DecisionMaker]):
            try:
                self.audio_stream = AudioStream()
                self.stt = SpeechToText()
                self.text_processor = TextProcessor()
                self.classifier = EmergencyClassifier()
                self.llm = LLMReasoner()
                self.decision_maker = DecisionMaker()
                self.initialized = True
                print("✓ Emergency Detection System initialized successfully")
            except Exception as e:
                print(f"⚠️  Could not initialize emergency system: {type(e).__name__}: {str(e)[:50]}")
                self.initialized = False
        else:
            print("⚠️  Emergency detection modules not fully available - running in demo mode")
            self.initialized = False
    
    def start_monitoring(self):
        """Start emergency detection monitoring"""
        if not self.initialized:
            return False
        
        try:
            self.monitoring = True
            self.audio_stream.start_stream()
            log_system_event("MONITORING_STARTED", "Emergency monitoring started", "Active")
            return True
        except Exception as e:
            log_system_event("ERROR", f"Failed to start monitoring: {str(e)}", "Error")
            self.monitoring = False
            return False
    
    def stop_monitoring(self):
        """Stop emergency detection monitoring"""
        if not self.initialized:
            return False
        
        try:
            self.monitoring = False
            self.audio_stream.stop_stream()
            log_system_event("MONITORING_STOPPED", "Emergency monitoring stopped", "Inactive")
            return True
        except Exception as e:
            log_system_event("ERROR", f"Failed to stop monitoring: {str(e)}", "Error")
            return False
    
    def process_audio_chunk(self):
        """Process a single audio chunk"""
        if not self.initialized or not self.monitoring:
            return None
        
        try:
            chunk = self.audio_stream.get_chunk()
            if chunk is not None:
                # Transcribe audio
                transcription = self.stt.transcribe(chunk)
                
                if transcription:
                    # Process and classify
                    cleaned_text = self.text_processor.clean_text(transcription)
                    confidence = self.classifier.predict(cleaned_text)
                    
                    # Get LLM reasoning
                    llm_response = self.llm.reason(transcription, confidence)
                    
                    # Make decision
                    decision = self.decision_maker.decide(confidence, llm_response, cleaned_text)
                    
                    # Log to database
                    is_emergency = 1 if decision == "Emergency" else 0
                    log_emergency_event(transcription, confidence, decision, is_emergency)
                    
                    return {
                        'transcription': transcription,
                        'confidence': round(confidence, 4),
                        'decision': decision,
                        'is_emergency': is_emergency,
                        'llm_response': llm_response,
                        'timestamp': datetime.now().isoformat()
                    }
        except Exception as e:
            log_system_event("ERROR", f"Error processing audio: {str(e)}", "Error")
        
        return None

# Initialize emergency detection system
try:
    emergency_system = EmergencyDetectionSystem()
    system_initialized = emergency_system.initialized
except:
    system_initialized = False
    emergency_system = None

# Helper functions
def log_emergency_event(transcription, confidence, decision, is_emergency):
    """Log emergency event to database"""
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO emergency_logs (audio_transcription, confidence, decision, is_emergency)
        VALUES (?, ?, ?, ?)
    ''', (transcription, confidence, decision, is_emergency))
    conn.commit()
    conn.close()
    
    # Update global stats
    system_status['events_count'] += 1
    system_status['last_event'] = datetime.now().isoformat()

def log_system_event(event_type, message, status):
    """Log system event to database"""
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO system_logs (event_type, message, status)
        VALUES (?, ?, ?)
    ''', (event_type, message, status))
    conn.commit()
    conn.close()

def monitoring_loop():
    """Main monitoring loop running in background thread"""
    global monitoring_active, system_status
    
    while monitoring_active and system_initialized and emergency_system:
        try:
            result = emergency_system.process_audio_chunk()
            if result:
                emergency_events.append(result)
                # Keep only last 100 events
                if len(emergency_events) > 100:
                    emergency_events.pop(0)
                
                system_status['monitoring'] = True
        except Exception as e:
            log_system_event("ERROR", f"Monitoring loop error: {str(e)}", "Error")
        
        time.sleep(0.1)  # Small delay to prevent CPU overuse

# Routes
@app.route('/')
def index():
    try:
        conn = get_db_connection()
        
        # Get regular events
        events = conn.execute('SELECT * FROM events ORDER BY date_time DESC LIMIT 10').fetchall()
        
        # Get emergency logs
        emergency_logs = conn.execute('''
            SELECT * FROM emergency_logs 
            WHERE is_emergency = 1 
            ORDER BY timestamp DESC LIMIT 5
        ''').fetchall()
        
        # Get system status
        total_events = conn.execute('SELECT COUNT(*) as count FROM events').fetchone()['count']
        active_events = conn.execute("SELECT COUNT(*) as count FROM events WHERE status='Active'").fetchone()['count']
        emergency_count = conn.execute("SELECT COUNT(*) as count FROM emergency_logs WHERE is_emergency = 1").fetchone()['count']
        
        conn.close()
        
        return render_template('index.html', 
                             events=events, 
                             total_events=total_events, 
                             active_events=active_events,
                             emergency_count=emergency_count,
                             emergency_logs=emergency_logs,
                             monitoring_active=system_status['monitoring'])
    except Exception as e:
        return render_template('index.html', events=[], error=str(e))

@app.route('/add-event', methods=['GET', 'POST'])
def add_event():
    if request.method == 'POST':
        try:
            event_name = request.form.get('event_name')
            event_type = request.form.get('event_type')
            description = request.form.get('description')
            location = request.form.get('location')
            confidence = float(request.form.get('confidence', 0))
            
            conn = get_db_connection()
            conn.execute(
                'INSERT INTO events (event_name, event_type, description, location, confidence) VALUES (?, ?, ?, ?, ?)',
                (event_name, event_type, description, location, confidence)
            )
            conn.commit()
            conn.close()
            return redirect(url_for('index'))
        except Exception as e:
            return render_template('add_event.html', error=str(e))
    
    return render_template('add_event.html')

@app.route('/events')
def view_events():
    try:
        conn = get_db_connection()
        events = conn.execute('SELECT * FROM events ORDER BY date_time DESC').fetchall()
        conn.close()
        return render_template('events.html', events=events)
    except Exception as e:
        return render_template('events.html', events=[], error=str(e))

@app.route('/event/<int:event_id>')
def event_detail(event_id):
    try:
        conn = get_db_connection()
        event = conn.execute('SELECT * FROM events WHERE id = ?', (event_id,)).fetchone()
        conn.close()
        if event is None:
            return redirect(url_for('view_events'))
        return render_template('event_detail.html', event=event)
    except Exception as e:
        return redirect(url_for('view_events'))

@app.route('/edit-event/<int:event_id>', methods=['GET', 'POST'])
def edit_event(event_id):
    try:
        conn = get_db_connection()
        event = conn.execute('SELECT * FROM events WHERE id = ?', (event_id,)).fetchone()
        
        if request.method == 'POST':
            event_name = request.form.get('event_name')
            event_type = request.form.get('event_type')
            description = request.form.get('description')
            location = request.form.get('location')
            status = request.form.get('status')
            confidence = float(request.form.get('confidence', 0))
            
            conn.execute(
                'UPDATE events SET event_name = ?, event_type = ?, description = ?, location = ?, status = ?, confidence = ? WHERE id = ?',
                (event_name, event_type, description, location, status, confidence, event_id)
            )
            conn.commit()
            conn.close()
            return redirect(url_for('event_detail', event_id=event_id))
        
        conn.close()
        return render_template('edit_event.html', event=event)
    except Exception as e:
        return render_template('edit_event.html', error=str(e))

@app.route('/delete-event/<int:event_id>', methods=['POST'])
def delete_event(event_id):
    try:
        conn = get_db_connection()
        conn.execute('DELETE FROM events WHERE id = ?', (event_id,))
        conn.commit()
        conn.close()
        return redirect(url_for('view_events'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/statistics')
def statistics():
    try:
        conn = get_db_connection()
        total_events = conn.execute('SELECT COUNT(*) FROM events').fetchone()[0]
        active_events = conn.execute("SELECT COUNT(*) FROM events WHERE status='Active'").fetchone()[0]
        
        event_types = conn.execute(
            'SELECT event_type, COUNT(*) as count FROM events GROUP BY event_type ORDER BY count DESC'
        ).fetchall()
        
        avg_confidence = conn.execute('SELECT AVG(confidence) FROM events').fetchone()[0]
        conn.close()
        
        return render_template('statistics.html', 
                             total_events=total_events,
                             active_events=active_events,
                             event_types=event_types,
                             avg_confidence=round(avg_confidence, 2) if avg_confidence else 0)
    except Exception as e:
        return render_template('statistics.html', error=str(e))

@app.route('/api/events')
def api_events():
    try:
        conn = get_db_connection()
        events = conn.execute('SELECT * FROM events ORDER BY date_time DESC').fetchall()
        conn.close()
        return jsonify([dict(e) for e in events])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== EMERGENCY DETECTION ROUTES ====================

@app.route('/emergency-dashboard')
def emergency_dashboard():
    """Main emergency detection dashboard"""
    try:
        conn = get_db_connection()
        
        # Get emergency statistics
        total_emergencies = conn.execute(
            "SELECT COUNT(*) as count FROM emergency_logs WHERE is_emergency = 1"
        ).fetchone()['count']
        
        recent_logs = conn.execute('''
            SELECT * FROM emergency_logs 
            ORDER BY timestamp DESC LIMIT 20
        ''').fetchall()
        
        system_logs = conn.execute('''
            SELECT * FROM system_logs 
            ORDER BY timestamp DESC LIMIT 10
        ''').fetchall()
        
        # Get average confidence
        avg_conf = conn.execute(
            "SELECT AVG(confidence) as avg FROM emergency_logs"
        ).fetchone()['avg']
        avg_confidence = round(avg_conf, 2) if avg_conf else 0
        
        conn.close()
        
        return render_template('emergency_dashboard.html',
                             total_emergencies=total_emergencies,
                             recent_logs=recent_logs,
                             system_logs=system_logs,
                             avg_confidence=avg_confidence,
                             monitoring_active=system_status['monitoring'])
    except Exception as e:
        return render_template('emergency_dashboard.html', error=str(e))

@app.route('/start-monitoring', methods=['POST'])
def start_monitoring():
    """Start emergency detection monitoring"""
    global monitoring_active, monitor_thread, system_status
    
    try:
        if not system_initialized or emergency_system is None:
            return jsonify({
                'success': False, 
                'message': 'Emergency detection system not initialized'
            }), 500
        
        if monitoring_active:
            return jsonify({'success': False, 'message': 'Monitoring already active'})
        
        # Start emergency detection system
        if emergency_system.start_monitoring():
            monitoring_active = True
            system_status['monitoring'] = True
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
            monitor_thread.start()
            
            return jsonify({
                'success': True,
                'message': 'Emergency monitoring started',
                'status': system_status
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to start monitoring'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stop-monitoring', methods=['POST'])
def stop_monitoring():
    """Stop emergency detection monitoring"""
    global monitoring_active, system_status
    
    try:
        if not monitoring_active:
            return jsonify({'success': False, 'message': 'Monitoring not active'})
        
        monitoring_active = False
        system_status['monitoring'] = False
        
        if emergency_system:
            emergency_system.stop_monitoring()
        
        return jsonify({
            'success': True,
            'message': 'Emergency monitoring stopped',
            'status': system_status
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/emergency-logs')
def api_emergency_logs():
    """Get emergency logs as JSON"""
    try:
        conn = get_db_connection()
        
        limit = request.args.get('limit', 50, type=int)
        emergency_only = request.args.get('emergency_only', False, type=bool)
        
        query = 'SELECT * FROM emergency_logs'
        params = []
        
        if emergency_only:
            query += ' WHERE is_emergency = 1'
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        logs = conn.execute(query, params).fetchall()
        conn.close()
        
        return jsonify([dict(log) for log in logs])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-status')
def api_system_status():
    """Get current system status"""
    try:
        conn = get_db_connection()
        
        total_emergencies = conn.execute(
            "SELECT COUNT(*) as count FROM emergency_logs WHERE is_emergency = 1"
        ).fetchone()['count']
        
        recent_emergency = conn.execute(
            "SELECT timestamp FROM emergency_logs WHERE is_emergency = 1 ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        
        conn.close()
        
        return jsonify({
            'monitoring_active': system_status['monitoring'],
            'total_emergencies': total_emergencies,
            'events_processed': system_status['events_count'],
            'last_event': system_status['last_event'],
            'recent_emergency': recent_emergency['timestamp'] if recent_emergency else None,
            'system_initialized': system_initialized
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emergency-events')
def api_emergency_events():
    """Get real-time emergency events (for WebSocket-like polling)"""
    return jsonify(emergency_events[-10:])  # Return last 10 events

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Live Event Detection System - Flask Web Application")
    print("=" * 60)
    print(f"✓ Emergency Detection System: {'Initialized' if system_initialized else 'Not Available'}")
    print(f"✓ Database: {app.config['DATABASE']}")
    print(f"✓ Debug Mode: On")
    print("=" * 60)
    print("\n📍 Routes Available:")
    print("   → Home: http://localhost:5000/")
    print("   → Emergency Dashboard: http://localhost:5000/emergency-dashboard")
    print("   → Events Management: http://localhost:5000/events")
    print("   → Statistics: http://localhost:5000/statistics")
    print("   → API: http://localhost:5000/api/events")
    print("\n⚠️  NOTE: Emergency detection requires audio hardware and proper module setup")
    print("=" * 60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
