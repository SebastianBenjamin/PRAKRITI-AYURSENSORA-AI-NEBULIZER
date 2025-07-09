import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.1/firebase-app.js";
import { getDatabase } from "https://www.gstatic.com/firebasejs/9.22.1/firebase-database.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/9.22.1/firebase-auth.js";

const firebaseConfig = {
    apiKey: "AIzaSyAgPbK7eGIWpq9Fe9IuP8lu6AslkXjSw-g",
    authDomain: "prakriti-ml.firebaseapp.com",
    databaseURL: "https://prakriti-ml-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "prakriti-ml",
    storageBucket: "prakriti-ml.firebasestorage.app",
    messagingSenderId: "228591322840",
    appId: "1:228591322840:web:5ca85186487477c2cd39ad",
    measurementId: "G-400WKJ325J"
};

const app = initializeApp(firebaseConfig);
const database = getDatabase(app);
const auth = getAuth(app);

export { app, database, auth };