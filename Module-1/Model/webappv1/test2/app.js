import { database, auth } from './firebase-config.js';
import { ref, set, get, update, remove } from "https://www.gstatic.com/firebasejs/9.22.1/firebase-database.js";
import { createUserWithEmailAndPassword, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/9.22.1/firebase-auth.js";

// Dosha Quiz Questions
const doshaquestions = [
    {
        question: "How would you describe your general body type?",
        options: [
            { text: "Lean and athletic, with a tendency to lose weight easily.", dosha: "Vata" },
            { text: "Medium build, muscular, with a tendency to stay fit.", dosha: "Pitta" },
            { text: "Sturdy, broad, and may gain weight easily.", dosha: "Kapha" }
        ]
    },
    // ... (rest of the 10 questions)
];

// Pages
const loginPage = document.getElementById('login-page');
const registerPage = document.getElementById('register-page');
const dashboardPage = document.getElementById('dashboard-page');

// Forms
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');

// Buttons
const analyzeBtn = document.getElementById('analyze-btn');
const registerBtn = document.getElementById('register-btn');
const logoutBtn = document.getElementById('logout-btn');
const showRegisterLink = document.getElementById('show-register');

// Modal
const modal = document.getElementById('modal');
const modalMessage = document.getElementById('modal-message');
const closeModal = document.querySelector('.close-modal');

// Dashboard Elements
const deviceStatusToggle = document.getElementById('device-status-toggle');
const sprayDelayInput = document.getElementById('spray-delay');
const sprayNowBtn = document.getElementById('spray-now-btn');
const userDoshaElement = document.getElementById('user-dosha');
const lastUpdateTimeElement = document.getElementById('last-update-time');

// Utility Functions
function showModal(message) {
    modalMessage.textContent = message;
    modal.style.display = 'block';
}

function hideModal() {
    modal.style.display = 'none';
}

closeModal.onclick = hideModal;
window.onclick = (event) => {
    if (event.target == modal) {
        hideModal();
    }
}

// Authentication Functions
async function registerUser(email, password, age, dosha) {
    try {
        // Check if email already exists
        const usersRef = ref(database, 'Users');
        const snapshot = await get(usersRef);
        const users = snapshot.val() || {};

        const emailExists = Object.values(users).some(user => user.UserEmail === email);
        if (emailExists) {
            showModal('Email already registered. Please login.');
            return false;
        }

        // Create user in Firebase Authentication
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        const userId = userCredential.user.uid;

        // Store user details in Realtime Database
        const userRef = ref(database, `Users/${userId}`);
        await set(userRef, {
            UserEmail: email,
            UserPassword: password,
            Age: age,
            Dosha: dosha
        });

        showModal('Registration Successful!');
        return true;
    } catch (error) {
        showModal(`Registration Error: ${error.message}`);
        return false;
    }
}

async function loginUser(deviceNumber, email, password) {
    try {
        // Check device availability
        const deviceRef = ref(database, `Devices/${deviceNumber}`);
        const deviceSnapshot = await get(deviceRef);
        const deviceData = deviceSnapshot.val();

        if (deviceData && deviceData.User) {
            showModal('Device already allocated to another user');
            return false;
        }

        // Authenticate user
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        const userId = userCredential.user.uid;

        // Update device allocation
        await update(deviceRef, {
            User: userId,
            Status: 'Active'
        });

        // Store login details
        localStorage.setItem('loggedInUser', JSON.stringify({
            userId,
            email,
            deviceNumber
        }));

        return true;
    } catch (error) {
        showModal(`Login Error: ${error.message}`);
        return false;
    }
}

// Page Navigation
showRegisterLink.addEventListener('click', () => {
    loginPage.classList.remove('active');
    registerPage.classList.add('active');
});

// Registration Flow
let userDosha = null;
analyzeBtn.addEventListener('click', () => {
    // Implement dosha analysis logic here
    // For now, a simple mock implementation
    userDosha = 'Vata'; // or you can add a more complex dosha determination
    const doshaSummary = {
        Vata: 35,
        Pitta: 30,
        Kapha: 35
    };

    const ctx = document.getElementById('dosha-chart').getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(doshaSummary),
            datasets: [{
                data: Object.values(doshaSummary),
                backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56']
            }]
        }
    });

    document.getElementById('dosha-chart-container').style.display = 'block';
    analyzeBtn.style.display = 'none';
    registerBtn.style.display = 'block';
});

registerForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('register-email').value;
    const password = document.getElementById('register-password').value;
    const age = document.getElementById('register-age').value;

    if (userDosha && await registerUser(email, password, age, userDosha)) {
        registerPage.classList.remove('active');
        loginPage.classList.add('active');
    }
});

// Login Flow
loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const deviceNumber = document.getElementById('login-device-number').value;
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;

    if (await loginUser(deviceNumber, email, password)) {
        loginPage.classList.remove('active');
        dashboardPage.classList.add('active');
        loadDashboard();
    }
});

// Dashboard Management
function loadDashboard() {
    const loggedInUser = JSON.parse(localStorage.getItem('loggedInUser'));
    if (loggedInUser) {
        const userRef = ref(database, `Users/${loggedInUser.userId}`);
        get(userRef).then((snapshot) => {
            const userData = snapshot.val();
            userDoshaElement.textContent = userData.Dosha;
        });
    }
}

logoutBtn.addEventListener('click', async () => {
    const loggedInUser = JSON.parse(localStorage.getItem('loggedInUser'));
    if (loggedInUser) {
        // Remove device allocation
        const deviceRef = ref(database, `Devices/${loggedInUser.deviceNumber}`);
        await update(deviceRef, { User: null, Status: 'Inactive' });

        // Clear local storage and redirect to login
        localStorage.removeItem('loggedInUser');
        dashboardPage.classList.remove('active');
        loginPage.classList.add('active');
    }
});

// Device Status and Spray Control
deviceStatusToggle.addEventListener('change', async () => {
    const loggedInUser = JSON.parse(localStorage.getItem('loggedInUser'));
    if (loggedInUser) {
        const deviceRef = ref(database, `Devices/${loggedInUser.deviceNumber}`);
        await update(deviceRef, {
            Status: deviceStatusToggle.checked ? 'Active' : 'Inactive'
        });
    }
});

sprayDelayInput.addEventListener('change', async () => {
    const loggedInUser = JSON.parse(localStorage.getItem('loggedInUser'));
    if (loggedInUser) {
        const deviceRef = ref(database, `Devices/${loggedInUser.deviceNumber}`);
        const delay = parseInt(sprayDelayInput.value);
        
        if (delay >= 1 && delay <= 100) {
            await update(deviceRef, {
                SprayDelay: delay,
                LastUpdate: new Date().toISOString()
            });
            lastUpdateTimeElement.textContent = new Date().toLocaleString();
        } else {
            showModal('Spray delay must be between 1 and 100 minutes');
        }
    }
});

sprayNowBtn.addEventListener('click', async () => {
    const loggedInUser = JSON.parse(localStorage.getItem('loggedInUser'));
    if (loggedInUser) {
        const deviceRef = ref(database, `Devices/${loggedInUser.deviceNumber}`);
        
        await update(deviceRef, {
            Spray: 1,
            LastUpdate: new Date().toISOString()
        });
        
        sprayNowBtn.classList.add('active');
        sprayNowBtn.classList.remove('inactive');
        lastUpdateTimeElement.textContent = new Date().toLocaleString();
        
        // Reset spray after a short delay
        setTimeout(async () => {
            await update(deviceRef, {
                Spray: 0
            });
            sprayNowBtn.classList.remove('active');
            sprayNowBtn.classList.add('inactive');
        }, 5000); // Reset after 5 seconds
    }
});

// Dosha Recheck
document.getElementById('recheck-dosha-dashboard').addEventListener('click', () => {
    // Reset dosha analysis form and show analysis options
    registerPage.classList.add('active');
    dashboardPage.classList.remove('active');
    analyzeBtn.style.display = 'block';
    registerBtn.style.display = 'none';
    document.getElementById('dosha-chart-container').style.display = 'none';
    userDosha = null;
});

// Page Load Authentication Check
document.addEventListener('DOMContentLoaded', () => {
    const loggedInUser = localStorage.getItem('loggedInUser');
    if (loggedInUser) {
        loginPage.classList.remove('active');
        dashboardPage.classList.add('active');
        loadDashboard();
    }
});