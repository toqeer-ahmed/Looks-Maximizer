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
    console.log("Firebase config not found. Running in Local Backend mode.");
  }
} catch (error) {
  console.log("Running in Local Backend mode (Firebase init skipped).");
}


const appId = (typeof __app_id !== 'undefined') ? __app_id : 'looks-maximizer-mvp';

// --- API Configuration ---
// Use Railway backend in production, localhost in development
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

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

const PremiumModal = ({ isOpen, onClose, onUpgrade }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in overflow-y-auto">
      <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 max-w-4xl w-full relative shadow-2xl shadow-purple-500/20 my-8">
        <button onClick={onClose} className="absolute top-4 right-4 text-slate-400 hover:text-white z-10">
          <X className="w-6 h-6" />
        </button>

        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold mb-2">Maximize Your Potential</h2>
          <p className="text-slate-400">Choose the plan that fits your goals.</p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Pro Plan */}
          <div className="border border-purple-500/30 bg-slate-900/50 rounded-xl p-6 relative hover:border-purple-500 transition-colors">
            <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-purple-600 px-3 py-1 rounded-full text-xs font-bold shadow-lg">MOST POPULAR</div>
            <h3 className="text-xl font-bold mb-2">Pro</h3>
            <div className="text-3xl font-bold mb-4">Rs 500<span className="text-sm text-slate-500 font-normal">/mo</span></div>
            <p className="text-sm text-slate-400 mb-6">For individuals serious about looks maxing.</p>
            <ul className="space-y-3 mb-8 text-sm">
              {[
                "Unlimited AI Analysis",
                "Full Looks Maxer Report (PDF)",
                "Detailed Symmetry Analysis",
                "Skin Quality Indicators",
                "Priority Inference Queue"
              ].map((item, i) => (
                <li key={i} className="flex items-center gap-2 text-slate-300">
                  <CheckCircle className="w-4 h-4 text-purple-400 shrink-0" /> {item}
                </li>
              ))}
            </ul>
            <Button onClick={() => onUpgrade('pro')} className="w-full bg-purple-600 hover:bg-purple-500">Get Pro</Button>
          </div>

          {/* Elite Plan */}
          <div className="border border-amber-500/30 bg-slate-900/50 rounded-xl p-6 relative hover:border-amber-500 transition-colors">
            <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-gradient-to-r from-amber-500 to-orange-500 px-3 py-1 rounded-full text-xs font-bold shadow-lg text-black">ELITE</div>
            <h3 className="text-xl font-bold mb-2 text-amber-500">Elite</h3>
            <div className="text-3xl font-bold mb-4">Rs 1000<span className="text-sm text-slate-500 font-normal">/mo</span></div>
            <p className="text-sm text-slate-400 mb-6">For influencers & content creators.</p>
            <ul className="space-y-3 mb-8 text-sm">
              {[
                "Everything in Pro",
                "Side-by-side Comparisons",
                "Style Evolution Tracking",
                "AI Improvement Roadmap",
                "Professional Grooming Tips"
              ].map((item, i) => (
                <li key={i} className="flex items-center gap-2 text-slate-300">
                  <CheckCircle className="w-4 h-4 text-amber-500 shrink-0" /> {item}
                </li>
              ))}
            </ul>
            <Button onClick={() => onUpgrade('elite')} className="w-full bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white">Get Elite</Button>
          </div>
        </div>

        <div className="mt-8 bg-slate-900/50 p-6 rounded-xl border border-blue-500/30">
          <h3 className="text-xl font-bold mb-4 text-center text-blue-400">Manual Payment Instructions</h3>
          <p className="text-sm text-slate-400 text-center mb-6">
            Please send the amount to one of the accounts below and <strong>send a screenshot</strong> to the respective contact to activate your plan.
          </p>
          <div className="grid gap-4 md:grid-cols-3 text-sm">
            <div className="bg-slate-800 p-4 rounded-lg text-center">
              <div className="font-bold text-white mb-1">Toqeer Ahmed</div>
              <div className="text-slate-300 font-mono">03339200251</div>
              <div className="text-orange-400 font-semibold mt-1">Sadapay</div>
            </div>
            <div className="bg-slate-800 p-4 rounded-lg text-center">
              <div className="font-bold text-white mb-1">Mustajaab</div>
              <div className="text-slate-300 font-mono">03136725787</div>
              <div className="text-green-400 font-semibold mt-1">Nayapay</div>
            </div>
            <div className="bg-slate-800 p-4 rounded-lg text-center">
              <div className="font-bold text-white mb-1">Hiba</div>
              <div className="text-slate-300 font-mono text-xs">19797901102899</div>
              <div className="text-red-400 font-semibold mt-1">HBL</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// --- Views ---

const LandingView = ({ onGetStarted }) => (
  <div className="min-h-screen flex flex-col">
    <nav className="p-6 flex justify-between items-center max-w-7xl mx-auto w-full">
      <div className="flex items-center gap-2">
        <div className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-violet-400">
          Looks Maxer
        </div>
        <span className="bg-purple-500/10 text-purple-400 text-xs px-2 py-0.5 rounded-full border border-purple-500/20 font-medium">BETA</span>
      </div>
      <div className="flex gap-4">
        <button className="hidden md:block text-slate-400 hover:text-white transition-colors">Pricing</button>
        <Button variant="secondary" onClick={onGetStarted} className="!px-4 !py-2 text-sm">Sign In</Button>
      </div>
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

      {/* Pricing Section */}
      {/* Pricing Section */}
      <div className="mt-32 max-w-6xl mx-auto w-full text-center pb-20">
        <h2 className="text-3xl font-bold mb-4">Pricing Plans</h2>
        <p className="text-slate-400 mb-12">Start for free, upgrade for mastery.</p>

        <div className="grid md:grid-cols-3 gap-8 px-4">
          {/* Free Plan */}
          <Card className="text-left border-slate-700 bg-slate-900/50 flex flex-col">
            <h3 className="text-xl font-bold mb-2">Starter</h3>
            <div className="text-3xl font-bold mb-6">$0<span className="text-lg text-slate-500 font-normal">/mo</span></div>
            <ul className="space-y-4 mb-8 flex-1">
              <li className="flex items-center gap-3 text-slate-300"><CheckCircle className="w-5 h-5 text-slate-500" /> 1 Photo per Day</li>
              <li className="flex items-center gap-3 text-slate-300"><CheckCircle className="w-5 h-5 text-slate-500" /> Basic Analysis</li>
              <li className="flex items-center gap-3 text-slate-300"><CheckCircle className="w-5 h-5 text-slate-500" /> 1 Recommendation</li>
            </ul>
            <Button variant="outline" onClick={onGetStarted} className="w-full mt-auto">Get Started</Button>
          </Card>

          {/* Pro Plan */}
          <Card className="text-left border-purple-500/50 bg-gradient-to-b from-slate-800 to-slate-900 relative overflow-hidden flex flex-col transform scale-105 shadow-2xl shadow-purple-900/20 z-10">
            <div className="absolute top-0 right-0 bg-purple-500 text-xs font-bold px-3 py-1 rounded-bl-lg">POPULAR</div>
            <h3 className="text-xl font-bold mb-2">Pro</h3>
            <div className="text-3xl font-bold mb-6">Rs 500<span className="text-lg text-slate-500 font-normal">/mo</span></div>
            <ul className="space-y-4 mb-8 flex-1">
              <li className="flex items-center gap-3 text-white"><CheckCircle className="w-5 h-5 text-purple-400" /> <strong>Unlimited</strong> Uploads</li>
              <li className="flex items-center gap-3 text-white"><CheckCircle className="w-5 h-5 text-purple-400" /> Detailed Reports (PDF)</li>
              <li className="flex items-center gap-3 text-white"><CheckCircle className="w-5 h-5 text-purple-400" /> 3-5 Style Options</li>
              <li className="flex items-center gap-3 text-white"><CheckCircle className="w-5 h-5 text-purple-400" /> Symmetry Analysis</li>
            </ul>
            <Button onClick={onGetStarted} className="w-full bg-purple-600 hover:bg-purple-500 mt-auto">Start Free Trial</Button>
          </Card>

          {/* Elite Plan */}
          <Card className="text-left border-amber-500/30 bg-slate-900/50 flex flex-col">
            <h3 className="text-xl font-bold mb-2 text-amber-500">Elite</h3>
            <div className="text-3xl font-bold mb-6">Rs 1000<span className="text-lg text-slate-500 font-normal">/mo</span></div>
            <ul className="space-y-4 mb-8 flex-1">
              <li className="flex items-center gap-3 text-slate-300"><CheckCircle className="w-5 h-5 text-amber-500" /> Everything in Pro</li>
              <li className="flex items-center gap-3 text-slate-300"><CheckCircle className="w-5 h-5 text-amber-500" /> Before/After Comparisons</li>
              <li className="flex items-center gap-3 text-slate-300"><CheckCircle className="w-5 h-5 text-amber-500" /> Evolution Tracking</li>
              <li className="flex items-center gap-3 text-slate-300"><CheckCircle className="w-5 h-5 text-amber-500" /> Priority Support</li>
            </ul>
            <Button variant="outline" onClick={onGetStarted} className="w-full border-amber-500/50 text-amber-500 hover:bg-amber-500/10 mt-auto">Get Elite</Button>
          </Card>
        </div>
      </div>

    </main>
  </div>
);

const AuthView = ({ onAuthSuccess }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  // New Onboarding Fields
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('Male');

  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleAuth = async (e) => {
    e.preventDefault();
    if (!auth) {
      if (!email || !password) {
        setError("Please enter both email and password to proceed.");
        setLoading(false);
        return;
      }

      setLoading(true);
      setError('');

      try {
        const endpoint = isLogin ? '/api/auth/login' : '/api/auth/signup';
        const payload = {
          email,
          password,
          details: !isLogin ? { name, age, gender } : {}
        };

        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.message || 'Authentication failed');
        }

        // Store user in local state
        // data.user comes from backend with uid, email, details
        onAuthSuccess(data.user);
      } catch (err) {
        console.error("Auth error:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
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
              <Input
                type="text"
                placeholder="Full Name"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
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

const DashboardView = ({ user, onStartAnalysis, history }) => {
  const [showPremiumModal, setShowPremiumModal] = useState(false);
  const FREE_LIMIT = 5;
  const usageCount = history.length;
  const isLimitReached = usageCount >= FREE_LIMIT;

  const handleStartClick = () => {
    if (isLimitReached) {
      setShowPremiumModal(true);
    } else {
      onStartAnalysis();
    }
  };

  const handleUpgrade = (plan) => {
    alert(`To upgrade to ${plan.toUpperCase()}, please send the payment to one of the accounts listed below and share the screenshot.`);
    // We keep the modal open so they can see the details
    // setShowPremiumModal(false); 
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <PremiumModal
        isOpen={showPremiumModal}
        onClose={() => setShowPremiumModal(false)}
        onUpgrade={handleUpgrade}
      />

      <header className="flex justify-between items-center mb-12">
        <div>
          <h1 className="text-3xl font-bold mb-2">Hello, <span className="text-blue-400">{user?.displayName || user?.email?.split('@')[0]}</span></h1>
          <div className="flex items-center gap-2">
            <span className="text-slate-400">Free Plan: {usageCount} / {FREE_LIMIT} analyses used</span>
            {isLimitReached && <span className="text-xs bg-red-500/20 text-red-400 px-2 py-1 rounded-full border border-red-500/20">Limit Reached</span>}
          </div>
        </div>
        <Button onClick={handleStartClick} variant={isLimitReached ? 'secondary' : 'primary'}>
          {isLimitReached ? <Sparkles className="w-5 h-5 text-purple-400" /> : <Plus className="w-5 h-5" />}
          {isLimitReached ? 'Unlock Premium' : 'Start New Analysis'}
        </Button>
      </header>

      <section>
        <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
          <Activity className="w-5 h-5 text-blue-400" /> Recent Activity
        </h2>

        {history.length === 0 ? (
          <div className="text-center py-20 border-2 border-dashed border-slate-700 rounded-2xl">
            <p className="text-slate-500 mb-4">No analyses yet.</p>
            <Button variant="outline" onClick={handleStartClick}>Analyze First Photo</Button>
          </div>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* New Analysis Card */}
            <button
              onClick={handleStartClick}
              className={`bg-slate-800/30 border-2 border-dashed border-slate-700 rounded-2xl p-6 flex flex-col items-center justify-center gap-4 hover:border-blue-500/50 hover:bg-slate-800/50 transition-all group h-full min-h-[200px] ${isLimitReached ? 'opacity-75' : ''}`}
            >
              <div className={`w-16 h-16 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform ${isLimitReached ? 'bg-purple-500/20' : 'bg-slate-700/50'}`}>
                {isLimitReached ? <Sparkles className="w-8 h-8 text-purple-400" /> : <Plus className="w-8 h-8 text-slate-400 group-hover:text-blue-400" />}
              </div>
              <span className="font-semibold text-slate-300 group-hover:text-white">
                {isLimitReached ? 'Unlock Unlimited' : 'New Analysis'}
              </span>
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
};

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

const ResultsView = ({ results, onReset, onFeedback, onBack, onShowPremium }) => {
  const [feedbackGiven, setFeedbackGiven] = useState(null);

  const handleFeedback = (isHelpful) => {
    setFeedbackGiven(isHelpful);
    onFeedback(isHelpful);
  };

  const isPremium = results.is_premium;
  const isPreview = results.preview_only;

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
        {results.is_trial && (
          <div className="bg-purple-500/20 border border-purple-500/50 text-purple-300 px-4 py-2 rounded-lg inline-block text-sm font-semibold animate-pulse">
            One-Time Free Premium Report Unlocked! 🎁
          </div>
        )}
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
          <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
            Detailed Analysis
            {isPreview && <span className="text-xs bg-slate-700 px-2 py-1 rounded text-slate-300 font-normal">Preview Mode</span>}
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card className="p-4 text-center">
              <p className="text-slate-400 text-sm">Age Group</p>
              <p className="font-semibold text-lg">{results.age_group}</p>
            </Card>
            <Card className="p-4 text-center">
              <p className="text-slate-400 text-sm">Gender</p>
              <p className="font-semibold text-lg">{results.gender}</p>
            </Card>
            <Card className="p-4 text-center relative overflow-hidden">
              {isPreview && results.skin_quality === 'LOCKED' && (
                <div className="absolute inset-0 bg-slate-900/90 backdrop-blur-sm flex flex-col items-center justify-center z-10 cursor-pointer" onClick={onShowPremium}>
                  <div className="bg-purple-600/20 p-2 rounded-full mb-1"><Sparkles className="w-4 h-4 text-purple-400" /></div>
                  <span className="text-xs font-bold text-purple-400">UNLOCK</span>
                </div>
              )}
              <p className="text-slate-400 text-sm">Skin Quality</p>
              <p className="font-semibold text-lg">{results.skin_quality || 'Unknown'}</p>
            </Card>
            <Card className="p-4 text-center relative overflow-hidden">
              {isPreview && results.symmetry_analysis === 'LOCKED' && (
                <div className="absolute inset-0 bg-slate-900/90 backdrop-blur-sm flex flex-col items-center justify-center z-10 cursor-pointer" onClick={onShowPremium}>
                  <div className="bg-purple-600/20 p-2 rounded-full mb-1"><Sparkles className="w-4 h-4 text-purple-400" /></div>
                  <span className="text-xs font-bold text-purple-400">UNLOCK</span>
                </div>
              )}
              <p className="text-slate-400 text-sm">Symmetry</p>
              <p className="font-semibold text-lg">{results.symmetry_analysis || 'Unknown'}</p>
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
            <Card key={i} className="h-full relative overflow-hidden">
              <h4 className="text-lg font-semibold mb-4 text-blue-400">{category.title}</h4>
              <ul className="space-y-3">
                {category.items.map((item, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                    <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
              {isPreview && (
                <div className="absolute bottom-0 left-0 right-0 h-1/2 bg-gradient-to-t from-slate-900 to-transparent flex items-end justify-center pb-4">
                  <span className="text-xs text-slate-400 flex items-center gap-1">
                    <Sparkles className="w-3 h-3" /> More ideas locked
                  </span>
                </div>
              )}
            </Card>
          ))}
        </div>
      </div>

      {isPreview && (
        <div className="bg-gradient-to-r from-purple-900/50 to-blue-900/50 border border-purple-500/30 rounded-2xl p-8 text-center mb-12">
          <h3 className="text-2xl font-bold mb-2">Unlock Your Best Self</h3>
          <p className="text-slate-300 mb-6 max-w-lg mx-auto">
            Get your full PDF report, symmetry analysis, skin quality score, and unlimited retries with Looks Maxer Pro.
          </p>
          <Button onClick={onShowPremium} className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-lg px-8">
            Unlock Full Report - $9.99
          </Button>
        </div>
      )}

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

        <Button onClick={onReset} size="lg" className="w-full max-w-xs" variant="outline">
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
  const [showPremiumModal, setShowPremiumModal] = useState(false);
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
    if (user) {
      if (firebaseInitialized && db) {
        // Real Mode: Fetch from Firestore
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
      } else {
        // Local Backend Mode: Fetch from API
        const fetchHistory = async () => {
          try {
            const response = await fetch(`${API_BASE_URL}/api/history?userId=${user.uid}`);
            if (response.ok) {
              const data = await response.json();
              setHistory(data.history.map(item => ({
                id: item.id || `local_${Math.random()}`, // Ensure ID exists
                ...item,
                onDelete: handleDeleteAnalysis
              })));
            }
          } catch (err) {
            console.error("Failed to fetch history:", err);
          }
        };
        fetchHistory();
        // Poll for updates every 5s since we don't have real-time listeners for local file
        const interval = setInterval(fetchHistory, 5000);
        return () => clearInterval(interval);
      }
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

  const handleUpgrade = async (plan) => {
    try {
      // Mock API call to upgrade
      await fetch(`${API_BASE_URL}/api/subscription/upgrade`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: user.uid, plan: plan })
      });
      alert(`Successfully upgraded to ${plan.charAt(0).toUpperCase() + plan.slice(1)}! You can now enjoy unlimited features.`);
      setShowPremiumModal(false);
      // Force refresh of history/profile if needed (simplified for MVP)
    } catch (e) {
      alert("Upgrade failed. Please try again.");
    }
  };

  const handleAnalysis = async (imageData) => {
    if (!user) return;
    setLoading(true);

    try {
      const uploadedImageURL = "mocked_url_for_mvp";

      const response = await fetch(`${API_BASE_URL}/api/analyze_face`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId: user.uid,
          imageData: imageData, // Send Base64 directly
          uploadedImageURL: "local_upload",
          userDetails: user.details || {}
        })
      });

      if (response.status === 403) {
        // Limit reached
        setShowPremiumModal(true);
        setLoading(false);
        return;
      }

      if (!response.ok) throw new Error('API analysis failed.');

      const data = await response.json();

      // Merge the ID from the backend into the results for tracking
      const resultsWithId = {
        ...data.results,
        analysisId: data.analysisId
      };

      setAnalysisResult(resultsWithId);
      setView('results');
    } catch (error) {
      console.error("Analysis error:", error);
      alert("Analysis failed. Please try again. " + error.message);
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
              handleAnalysis(img);
            }}
            onBack={() => setView('upload')}
          />
        );
      case 'results':
        return <ResultsView
          results={analysisResult}
          onReset={() => setView('upload')}
          onFeedback={handleFeedback}
          onBack={() => setView('dashboard')}
          onShowPremium={() => setShowPremiumModal(true)}
        />;
      default:
        return <LandingView onGetStarted={() => setView('auth')} />;
    }
  };

  return (
    <div className="min-h-screen bg-darker text-white font-sans selection:bg-blue-500/30">
      <PremiumModal
        isOpen={showPremiumModal}
        onClose={() => setShowPremiumModal(false)}
        onUpgrade={handleUpgrade}
      />

      {/* Global Navbar (except landing which has its own) */}
      {view !== 'landing' && (
        <nav className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
            <div
              className="font-bold text-xl bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-violet-400 cursor-pointer"
              onClick={() => setView(user ? 'dashboard' : 'landing')}
            >
              Looks Maxer
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
