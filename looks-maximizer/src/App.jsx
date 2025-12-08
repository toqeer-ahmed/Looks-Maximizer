import React, { useState, useEffect, useRef } from 'react';
import { initializeApp } from 'firebase/app';
import {
  getAuth,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  signOut,
  GoogleAuthProvider,
  signInWithPopup
} from 'firebase/auth';
import {
  getFirestore,
  doc,
  setDoc,
  addDoc,
  collection,
  onSnapshot,
  query,
  orderBy,
  serverTimestamp
} from 'firebase/firestore';
import {
  Camera,
  Upload,
  User,
  LogOut,
  Activity,
  CheckCircle,
  XCircle,
  ChevronRight,
  Sparkles,

  Menu,
  X,
  Trash2,
  Plus
} from 'lucide-react';

// --- Firebase Initialization ---
// Safely attempt to initialize Firebase. If it fails (e.g. missing config), we'll fall back to mock mode.
let auth, db;
let firebaseInitialized = false;

try {
  const firebaseConfig = (typeof __firebase_config !== 'undefined') ? __firebase_config : null;

  if (firebaseConfig && Object.keys(firebaseConfig).length > 0) {
    const app = initializeApp(firebaseConfig);
    auth = getAuth(app);
    db = getFirestore(app);
    firebaseInitialized = true;
  } else {
    console.warn("Firebase config not found. Running in offline/mock mode.");
  }
} catch (error) {
  console.error("Failed to initialize Firebase:", error);
}


const appId = (typeof __app_id !== 'undefined') ? __app_id : 'looks-maximizer-mvp';

// --- API Configuration ---
// Use Railway backend in production, localhost in development
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8080";

// --- Error Boundary ---
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-slate-900 text-white p-4">
          <div className="max-w-md bg-slate-800 p-6 rounded-lg border border-red-500/50">
            <h2 className="text-xl font-bold text-red-400 mb-2">Something went wrong</h2>
            <p className="text-slate-300 mb-4">The application encountered an error.</p>
            <pre className="bg-black/50 p-4 rounded text-xs font-mono overflow-auto max-h-40">
              {this.state.error?.toString()}
            </pre>
            <button
              onClick={() => window.location.reload()}
              className="mt-4 px-4 py-2 bg-blue-600 rounded hover:bg-blue-500 w-full"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// --- Mock Data ---
const MOCKED_RESULTS = {
  faceShape: "Square",
  lookScore: 92,
  recommendations: {
    hairstyles: [
      "Quiff (Adds height, lengthens face)",
      "Low Fade (Softens jawline)",
      "Crew Cut (Classic, balances features)"
    ],
    beardStyles: [
      "Stubble (Always safe)",
      "Circle Beard (Adds length to chin)"
    ],
    clothingStyle: [
      "Structured jackets (Sharp lines)",
      "V-neck shirts (Softens the neck/jaw transition)",
      "Bold patterns (Draws attention to the chest)"
    ]
  }
};

// --- Components ---

const Button = ({ children, onClick, variant = 'primary', className = '', disabled = false, type = 'button' }) => {
  const baseStyle = "px-6 py-3 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center gap-2";
  const variants = {
    primary: "bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 text-white shadow-lg shadow-blue-500/20",
    secondary: "bg-slate-800 hover:bg-slate-700 text-white border border-slate-700",
    outline: "border-2 border-slate-600 hover:border-slate-400 text-slate-300 hover:text-white",
    danger: "bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20"
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${baseStyle} ${variants[variant]} ${disabled ? 'opacity-50 cursor-not-allowed' : ''} ${className}`}
    >
      {children}
    </button>
  );
};

const Input = ({ type, placeholder, value, onChange, className = '' }) => (
  <input
    type={type}
    placeholder={placeholder}
    value={value}
    onChange={onChange}
    className={`w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all ${className}`}
  />
);

const Card = ({ children, className = '' }) => (
  <div className={`bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6 ${className}`}>
    {children}
  </div>
);

// --- Views ---

const LandingView = ({ onGetStarted }) => (
  <div className="min-h-screen flex flex-col">
    <nav className="p-6 flex justify-between items-center max-w-7xl mx-auto w-full">
      <div className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-violet-400">
        LooksMax
      </div>
      <Button variant="secondary" onClick={onGetStarted} className="!px-4 !py-2 text-sm">Sign In</Button>
    </nav>

    <main className="flex-1 flex flex-col items-center justify-center text-center px-4 relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-blue-500/20 rounded-full blur-[100px] -z-10 animate-pulse" />

      <h1 className="text-5xl md:text-7xl font-bold mb-6 tracking-tight">
        Maximize Your <span className="text-blue-500">Potential</span>
      </h1>
      <p className="text-xl text-slate-400 mb-10 max-w-2xl">
        AI-powered analysis to discover your best look. Get personalized recommendations for hairstyles, grooming, and fashion.
      </p>
      <Button onClick={onGetStarted} className="text-lg px-8 py-4">
        Try For Free <ChevronRight className="w-5 h-5" />
      </Button>

      <div className="mt-20 grid md:grid-cols-3 gap-8 max-w-5xl mx-auto text-left">
        {[
          { title: "AI Analysis", desc: "Instant facial feature detection and scoring." },
          { title: "Personalized", desc: "Tailored advice for your unique face shape." },
          { title: "Track Progress", desc: "Monitor your improvements over time." }
        ].map((feature, i) => (
          <Card key={i} className="hover:bg-slate-800/80 transition-colors">
            <Sparkles className="w-8 h-8 text-blue-400 mb-4" />
            <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
            <p className="text-slate-400">{feature.desc}</p>
          </Card>
        ))}
      </div>
    </main>
  </div>
);

const AuthView = ({ onAuthSuccess }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  // New Onboarding Fields
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('Male');
  const [height, setHeight] = useState('');
  const [skinTone, setSkinTone] = useState('Fair');

  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleAuth = async (e) => {
    e.preventDefault();
    if (!auth) {
      // Mock Auth for offline mode
      const mockUser = {
        uid: "mock_user_" + Math.floor(Math.random() * 1000),
        email: email || "mock@example.com",
        displayName: email ? email.split('@')[0] : "Guest User",
        // Store extra details in user object for now
        details: !isLogin ? { age, gender, height, skinTone } : {}
      };

      setLoading(true);
      setTimeout(() => {
        onAuthSuccess(mockUser);
        setLoading(false);
      }, 800);
      return;
    }
    // ... Real Firebase Auth logic would go here ...
  };

  const handleGoogleSignIn = async () => {
    // ... (Keep existing logic)
    if (!auth) {
      onAuthSuccess({ uid: "mock_google", email: "google@example.com", displayName: "Google User" });
      return;
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 relative py-10">
      <button
        onClick={() => window.location.reload()}
        className="absolute top-6 left-6 text-slate-400 hover:text-white flex items-center gap-2 transition-colors"
      >
        <ChevronRight className="w-5 h-5 rotate-180" /> Back
      </button>
      <Card className="w-full max-w-md p-8">
        <h2 className="text-3xl font-bold mb-2 text-center">{isLogin ? 'Welcome Back' : 'Create Profile'}</h2>
        <p className="text-slate-400 text-center mb-8">
          {isLogin ? 'Enter your details to access your dashboard' : 'Tell us about yourself for better recommendations'}
        </p>

        <form onSubmit={handleAuth} className="space-y-4">
          <Input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <Input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          {!isLogin && (
            <>
              <div className="grid grid-cols-2 gap-4">
                <Input
                  type="number"
                  placeholder="Age"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                />
                <select
                  value={gender}
                  onChange={(e) => setGender(e.target.value)}
                  className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <Input
                  type="text"
                  placeholder="Height (e.g. 5'10)"
                  value={height}
                  onChange={(e) => setHeight(e.target.value)}
                />
                <select
                  value={skinTone}
                  onChange={(e) => setSkinTone(e.target.value)}
                  className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="Fair">Fair</option>
                  <option value="Medium">Medium</option>
                  <option value="Olive">Olive</option>
                  <option value="Brown">Brown</option>
                  <option value="Dark">Dark</option>
                </select>
              </div>
            </>
          )}

          {error && <p className="text-red-400 text-sm bg-red-500/10 p-3 rounded-lg">{error}</p>}

          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? 'Processing...' : (isLogin ? 'Sign In' : 'Create Account')}
          </Button>
        </form>

        <div className="mt-6 text-center">
          <button
            onClick={() => setIsLogin(!isLogin)}
            className="text-slate-400 hover:text-white text-sm transition-colors"
          >
            {isLogin ? "New here? Create a Profile" : "Already have an account? Sign In"}
          </button>
        </div>
      </Card>
    </div>
  );
};

const DashboardView = ({ user, onStartAnalysis, history }) => (
  <div className="max-w-6xl mx-auto px-4 py-8">
    <header className="flex justify-between items-center mb-12">
      <div>
        <h1 className="text-3xl font-bold mb-2">Hello, <span className="text-blue-400">{user?.displayName || user?.email?.split('@')[0]}</span></h1>
        <p className="text-slate-400">Ready to maximize your look today?</p>
      </div>
      <Button onClick={onStartAnalysis}>
        <Plus className="w-5 h-5" /> Start New Analysis
      </Button>
    </header>

    <section>
      <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
        <Activity className="w-5 h-5 text-blue-400" /> Recent Activity
      </h2>

      {history.length === 0 ? (
        <div className="text-center py-20 border-2 border-dashed border-slate-700 rounded-2xl">
          <p className="text-slate-500 mb-4">No analyses yet.</p>
          <Button variant="outline" onClick={onStartAnalysis}>Analyze First Photo</Button>
        </div>
      ) : (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* New Analysis Card */}
          <button
            onClick={onStartAnalysis}
            className="bg-slate-800/30 border-2 border-dashed border-slate-700 rounded-2xl p-6 flex flex-col items-center justify-center gap-4 hover:border-blue-500/50 hover:bg-slate-800/50 transition-all group h-full min-h-[200px]"
          >
            <div className="w-16 h-16 bg-slate-700/50 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
              <Plus className="w-8 h-8 text-slate-400 group-hover:text-blue-400" />
            </div>
            <span className="font-semibold text-slate-300 group-hover:text-white">New Analysis</span>
          </button>

          {history.map((item) => (
            <Card key={item.id} className="relative group hover:border-blue-500/50 transition-colors">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirm('Are you sure you want to delete this analysis?')) item.onDelete(item.id);
                }}
                className="absolute top-4 right-4 p-2 bg-slate-900/50 rounded-full text-slate-400 hover:text-red-400 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 transition-all"
                title="Delete Analysis"
              >
                <Trash2 className="w-4 h-4" />
              </button>

              <div className="flex justify-between items-start mb-4">
                <span className="bg-blue-500/20 text-blue-300 text-xs px-2 py-1 rounded-full">
                  {item.timestamp?.seconds ? new Date(item.timestamp.seconds * 1000).toLocaleDateString() : 'Just now'}
                </span>
                <span className="font-bold text-lg text-green-400">{item.lookScore}/100</span>
              </div>
              <h3 className="font-semibold mb-1">{item.faceShape}</h3>
              <p className="text-sm text-slate-400 truncate">{item.recommendations?.hairstyles?.[0]}</p>
            </Card>
          ))}
        </div>
      )}
    </section>
  </div>
);

const CameraView = ({ onCapture, onBack }) => {
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert("Could not access camera. Please allow permissions.");
      onBack();
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
  };

  const capture = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      canvas.getContext('2d').drawImage(videoRef.current, 0, 0);
      const imageData = canvas.toDataURL('image/jpeg');
      onCapture(imageData);
      stopCamera();
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-black">
      <div className="relative w-full max-w-2xl aspect-video bg-slate-900 rounded-lg overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="w-full h-full object-cover"
        />
        <button
          onClick={() => { stopCamera(); onBack(); }}
          className="absolute top-4 right-4 p-2 bg-black/50 text-white rounded-full hover:bg-red-500/80 transition-colors"
        >
          <X className="w-6 h-6" />
        </button>
      </div>
      <div className="mt-8 flex gap-4">
        <Button onClick={capture} className="px-8">
          <Camera className="w-5 h-5" /> Capture Photo
        </Button>
      </div>
    </div>
  );
};

const UploadView = ({ onAnalyze, loading, onBack, onOpenCamera }) => {
  const [image, setImage] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const reader = new FileReader();
      reader.onload = (e) => setImage(e.target.result);
      reader.readAsDataURL(e.target.files[0]);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4">
      <div className="w-full max-w-2xl relative">
        <button
          onClick={onBack}
          className="absolute -top-12 left-0 text-slate-400 hover:text-white flex items-center gap-2 transition-colors"
        >
          <ChevronRight className="w-5 h-5 rotate-180" /> Back to Dashboard
        </button>

        <h2 className="text-3xl font-bold text-center mb-8">Upload Your Photo</h2>

        <div className="bg-slate-800/50 border-2 border-dashed border-slate-600 rounded-3xl p-10 text-center relative overflow-hidden group hover:border-blue-500/50 transition-all">
          {image ? (
            <div className="relative">
              <img src={image} alt="Preview" className="max-h-[400px] mx-auto rounded-lg shadow-2xl" />
              <button
                onClick={() => setImage(null)}
                className="absolute top-2 right-2 bg-black/50 p-2 rounded-full hover:bg-red-500 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          ) : (
            <div className="py-10">
              <div className="w-20 h-20 bg-slate-700/50 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform">
                <Upload className="w-10 h-10 text-slate-400 group-hover:text-blue-400" />
              </div>
              <p className="text-xl font-medium mb-2">Drop your image here</p>
              <p className="text-slate-400 text-sm mb-8">or click to browse</p>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <Button variant="secondary" className="pointer-events-none relative z-10">
                Select Photo
              </Button>
            </div>
          )}
        </div>

        <div className="mt-8 flex justify-center gap-4">
          <Button
            variant="outline"
            onClick={onOpenCamera}
            className="flex-1"
          >
            <Camera className="w-5 h-5" /> Open Live Camera
          </Button>
          <Button
            onClick={() => onAnalyze(image)}
            disabled={!image || loading}
            className="w-48"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Analyzing... (approx 5-10s)
              </span>
            ) : (
              <>Analyze Now <Sparkles className="w-4 h-4" /></>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
};

const ResultsView = ({ results, onReset, onFeedback, onBack }) => {
  const [feedbackGiven, setFeedbackGiven] = useState(null);

  const handleFeedback = (isHelpful) => {
    setFeedbackGiven(isHelpful);
    onFeedback(isHelpful);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-10">
      <div className="text-center mb-12 relative">
        <button
          onClick={onBack}
          className="absolute top-0 left-0 text-slate-400 hover:text-white flex items-center gap-2 transition-colors"
        >
          <ChevronRight className="w-5 h-5 rotate-180" /> Dashboard
        </button>
        <div className="inline-block p-1 rounded-full bg-gradient-to-r from-blue-500 to-violet-500 mb-6">
          <div className="bg-slate-900 rounded-full px-6 py-2 text-sm font-bold tracking-wider uppercase">
            Analysis Complete
          </div>
        </div>
        <h2 className="text-4xl font-bold mb-4">Your Results</h2>
      </div>

      <div className="grid md:grid-cols-2 gap-8 mb-12">
        <Card className="flex flex-col items-center justify-center text-center py-10 bg-gradient-to-b from-slate-800 to-slate-900">
          <div className="w-32 h-32 rounded-full border-4 border-blue-500 flex items-center justify-center mb-6 relative">
            <span className="text-4xl font-bold">{results.lookScore}</span>
            <div className="absolute -bottom-2 bg-blue-600 text-xs px-2 py-1 rounded">SCORE</div>
          </div>
          <h3 className="text-xl font-semibold text-slate-300">Overall Look Score</h3>
        </Card>

        <Card className="flex flex-col items-center justify-center text-center py-10">
          <div className="w-24 h-24 bg-slate-700/50 rounded-2xl flex items-center justify-center mb-6">
            <User className="w-12 h-12 text-violet-400" />
          </div>
          <h3 className="text-2xl font-bold mb-1">{results.faceShape}</h3>
          <p className="text-slate-400">Detected Face Shape</p>
        </Card>
      </div>

      {/* Detailed Analysis Section */}
      {results.age_group && (
        <div className="mb-12">
          <h3 className="text-2xl font-bold mb-6">Detailed Analysis</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card className="p-4 text-center">
              <p className="text-slate-400 text-sm">Age Group</p>
              <p className="font-semibold text-lg">{results.age_group}</p>
            </Card>
            <Card className="p-4 text-center">
              <p className="text-slate-400 text-sm">Gender</p>
              <p className="font-semibold text-lg">{results.gender_hf || results.gender_celeba}</p>
            </Card>
            <Card className="p-4 text-center">
              <p className="text-slate-400 text-sm">Race/Ethnicity</p>
              <p className="font-semibold text-lg">{results.race}</p>
            </Card>
            <Card className="p-4 text-center">
              <p className="text-slate-400 text-sm">Hair Style</p>
              <p className="font-semibold text-lg">{results.hair?.replace('_', ' ')}</p>
            </Card>
            <Card className="p-4 text-center">
              <p className="text-slate-400 text-sm">Beard</p>
              <p className="font-semibold text-lg">{results.beard?.replace('_', ' ')}</p>
            </Card>
            <Card className="p-4 text-center">
              <p className="text-slate-400 text-sm">Glasses</p>
              <p className="font-semibold text-lg">{results.glasses ? "Yes" : "No"}</p>
            </Card>
            <Card className="p-4 text-center">
              <p className="text-slate-400 text-sm">Smiling</p>
              <p className="font-semibold text-lg">{results.smiling ? "Yes" : "No"}</p>
            </Card>
            <Card className="p-4 text-center">
              <p className="text-slate-400 text-sm">Attractiveness (AI)</p>
              <p className="font-semibold text-lg">{(results.attractive_score_celeba * 100).toFixed(0)}%</p>
            </Card>
          </div>
        </div>
      )}

      <div className="space-y-6 mb-12">
        <h3 className="text-2xl font-bold mb-6">Recommendations</h3>

        <div className="grid md:grid-cols-3 gap-6">
          {[
            { title: "Hairstyles", items: results.recommendations.hairstyles },
            { title: "Beard Styles", items: results.recommendations.beardStyles },
            { title: "Clothing", items: results.recommendations.clothingStyle }
          ].map((category, i) => (
            <Card key={i} className="h-full">
              <h4 className="text-lg font-semibold mb-4 text-blue-400">{category.title}</h4>
              <ul className="space-y-3">
                {category.items.map((item, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                    <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </Card>
          ))}
        </div>
      </div>

      <div className="flex flex-col items-center gap-8 border-t border-slate-800 pt-10">
        <div className="text-center">
          <p className="mb-4 text-slate-400">Was this analysis helpful?</p>
          <div className="flex gap-4 justify-center">
            <button
              onClick={() => handleFeedback(true)}
              className={`p-3 rounded-full transition-all ${feedbackGiven === true ? 'bg-green-500 text-white' : 'bg-slate-800 hover:bg-slate-700'}`}
            >
              <CheckCircle className="w-6 h-6" />
            </button>
            <button
              onClick={() => handleFeedback(false)}
              className={`p-3 rounded-full transition-all ${feedbackGiven === false ? 'bg-red-500 text-white' : 'bg-slate-800 hover:bg-slate-700'}`}
            >
              <XCircle className="w-6 h-6" />
            </button>
          </div>
        </div>

        <Button onClick={onReset} size="lg" className="w-full max-w-xs">
          Try Another Photo
        </Button>
      </div>
    </div>
  );
};

// --- Main App Component ---

function App() {
  const [view, setView] = useState('landing');
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    if (!firebaseInitialized || !auth) return;

    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      if (currentUser && view === 'landing') {
        setView('dashboard');
      }
    });
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    if (user && firebaseInitialized && db) {
      const q = query(
        collection(db, `artifacts/${appId}/users/${user.uid}/analysis`),
        orderBy('timestamp', 'desc')
      );
      const unsubscribe = onSnapshot(q, (snapshot) => {
        setHistory(snapshot.docs.map(doc => ({
          id: doc.id,
          ...doc.data(),
          onDelete: handleDeleteAnalysis
        })));
      });
      return () => unsubscribe();
    }
  }, [user]);

  const handleDeleteAnalysis = async (analysisId) => {
    if (!user) return;
    try {
      const response = await fetch(`${API_BASE_URL}/api/analysis/${analysisId}?userId=${user.uid}`, {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error('Delete failed');
      // Snapshot listener will automatically update the UI
    } catch (error) {
      console.error("Error deleting analysis:", error);
      alert("Failed to delete analysis.");
    }
  };

  const handleAnalysis = async (imageData) => {
    if (!user) return;
    setLoading(true);

    try {
      // In a real app, we would upload the 'imageData' (base64) to storage here 
      // and get a URL. For this MVP, we'll send a mock URL or the base64 if needed,
      // but the backend expects 'uploadedImageURL'.
      const uploadedImageURL = "mocked_url_for_mvp";

      const response = await fetch(`${API_BASE_URL}/api/analyze_face`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId: user.uid,
          imageData: imageData, // Send Base64 directly
          uploadedImageURL: "local_upload"
        })
      });

      if (!response.ok) throw new Error('API analysis failed.');

      const data = await response.json();

      // Merge the ID from the backend into the results for tracking
      const resultsWithId = {
        ...data.results,
        id: data.analysisId
      };

      setAnalysisResult(resultsWithId);
      setView('results');

    } catch (error) {
      console.error("Error during analysis API call:", error);
      alert("Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (isHelpful) => {
    if (!user || !analysisResult?.id) return;

    try {
      const response = await fetch(`${API_BASE_URL}/api/log_feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId: user.uid,
          analysisId: analysisResult.id,
          helpful: isHelpful
        })
      });

      if (!response.ok) throw new Error('API feedback logging failed.');

      // Optional: Show success message
      // alert("Thanks for your feedback!");

    } catch (error) {
      console.error("Error logging feedback via API:", error);
    }
  };

  const handleLogout = async () => {
    if (auth) await signOut(auth);
    setUser(null);
    setView('landing');
  };

  // View Routing
  const renderView = () => {
    switch (view) {
      case 'landing':
        return <LandingView onGetStarted={() => setView(user ? 'dashboard' : 'auth')} />;
      case 'auth':
        return <AuthView onAuthSuccess={(mockUser) => {
          if (mockUser) setUser(mockUser);
          setView('dashboard');
        }} />;
      case 'dashboard':
        return <DashboardView user={user} onStartAnalysis={() => setView('upload')} history={history} />;
      case 'upload':
        return (
          <UploadView
            onAnalyze={handleAnalysis}
            loading={loading}
            onBack={() => setView('dashboard')}
            onOpenCamera={() => setView('camera')}
          />
        );
      case 'camera':
        return (
          <CameraView
            onCapture={(img) => {
              // We can pass the captured image directly to analysis or back to upload preview
              // Let's go to upload preview with the image pre-filled? 
              // Actually, UploadView state is local. Let's just analyze directly or pass it somehow.
              // Simpler: Analyze directly or modify UploadView to accept initialImage.
              // Let's modify UploadView to accept initialImage prop, but that requires lifting state.
              // For now, let's just analyze directly for speed.
              handleAnalysis(img);
            }}
            onBack={() => setView('upload')}
          />
        );
      case 'results':
        return <ResultsView results={analysisResult} onReset={() => setView('upload')} onFeedback={handleFeedback} onBack={() => setView('dashboard')} />;
      default:
        return <LandingView onGetStarted={() => setView('auth')} />;
    }
  };

  return (
    <div className="min-h-screen bg-darker text-white font-sans selection:bg-blue-500/30">
      {/* Global Navbar (except landing which has its own) */}
      {view !== 'landing' && (
        <nav className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
            <div
              className="font-bold text-xl bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-violet-400 cursor-pointer"
              onClick={() => setView(user ? 'dashboard' : 'landing')}
            >
              LooksMax
            </div>
            {user && (
              <div className="flex items-center gap-4">
                <button
                  onClick={handleLogout}
                  className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors px-3 py-2 rounded-lg hover:bg-slate-800"
                >
                  <LogOut className="w-5 h-5" />
                  <span>Logout</span>
                </button>
              </div>
            )}
          </div>
        </nav>
      )}

      {renderView()}
    </div>
  );
}

export default function AppWrapper() {
  return (
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  );
}
