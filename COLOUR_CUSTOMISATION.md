# Streamlit Colour Customisation Guide

## How to Change Colours in Your Streamlit App

### Method 1: Theme Configuration File (`.streamlit/config.toml`)

Located at: `.streamlit/config.toml`

```toml
[theme]
primaryColour = "#FF4B4B"           # Buttons, links, accents (Red for fraud alerts)
backgroundColour = "#0E1117"         # Main background (Dark)
secondaryBackgroundColour = "#262730" # Sidebar, widgets (Lighter dark)
textColour = "#FAFAFA"              # Text colour (Light)
font = "sans serif"                 # Font family
```

**Colour Schemes:**

#### Dark Theme (Current)

- Primary: `#FF4B4B` (Red)
- Background: `#0E1117` (Dark gray)
- Secondary: `#262730` (Lighter gray)
- Text: `#FAFAFA` (White)

#### Light Theme

- Primary: `#8338EC` (Purple)
- Background: `#FFFFFF` (White)
- Secondary: `#262730` (Light gray)
- Text: `#F0F2F6` (Dark gray)

#### Cyberpunk Theme

- Primary: `#FF00FF` (Magenta)
- Background: `#0A0E27` (Deep blue-black)
- Secondary: `#1A1F4D` (Dark blue)
- Text: `#00FFFF` (Cyan)

#### Finance Theme

- Primary: `#00AEEE` (Turquoise)
- Background: `#F5F5F5` (Light gray)
- Secondary: `#E6F7FB` (Pale aqua)
- Text: `#1B263B` (Dark navy slate)

### Method 2: Custom CSS Injection (More Control)

In your `streamlit_app.py`, use `st.markdown()` with `unsafe_allow_html=True`:

```python
st.markdown("""
    <style>
    /* Custom styling here */
    h1 {
        color: #FF4B4B;
        background: linear-gradient(90deg, rgba(255,75,75,0.1) 0%, rgba(61,213,109,0.1) 100%);
    }
    
    /* Fraud alert box */
    .fraud-alert {
        background-color: rgba(255, 75, 75, 0.2);
        border-left: 5px solid #FF4B4B;
        color: #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)
```

### Method 3: Inline HTML/CSS in Markdown

```python
st.markdown("""
    <div style="background-color: rgba(255,75,75,0.3); 
                padding: 1rem; 
                border-radius: 10px; 
                color: #FF4B4B;">
        🚨 FRAUD DETECTED
    </div>
""", unsafe_allow_html=True)
```

## Colour Palette for Fraud Detection

### Alert Colours

- **Fraud/Danger**: `#FF4B4B` (Red)
- **Legit/Success**: `#3DD56D` (Green)
- **Warning**: `#FFA500` (Orange)
- **Info**: `#00B4D8` (Blue)

### Background Colours

- **Dark Primary**: `#0E1117`
- **Dark Secondary**: `#262730`
- **Light Primary**: `#FFFFFF`
- **Light Secondary**: `#F0F2F6`

### Accent Colours

- **Purple Gradient**: `#667eea` → `#764ba2`
- **Blue Gradient**: `#00B4D8` → `#0077B6`
- **Red Gradient**: `#FF4B4B` → `#C41E3A`

## How to Apply Changes

### For Theme Config

1. Edit `.streamlit/config.toml`
2. Save the file
3. Restart Streamlit server: `Ctrl+C` then `streamlit run streamlit_app.py`

### For CSS Injection

1. Edit `streamlit_app.py`
2. Save the file
3. Streamlit auto-reloads (or click "Rerun" in browser)

## Useful Streamlit CSS Selectors

```css
/* Main title */
h1, h2, h3 { color: #FF4B4B; }

/* Buttons */
.stButton > button { background-color: #667eea; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #262730; }

/* Metrics */
[data-testid="stMetricValue"] { font-size: 2rem; }

/* Alert boxes */
.stAlert { border-radius: 10px; }

/* Expanders */
.streamlit-expanderHeader { background-color: rgba(255, 255, 255, 0.05); }

/* Code blocks */
code { background-color: rgba(255, 255, 255, 0.1); }

/* Dataframes */
[data-testid="stDataFrame"] { border: 2px solid #667eea; }
```

## Testing Your Colours

Use this Python snippet to preview colours:

```python
import streamlit as st

st.markdown("""
    <div style="background-color: #FF4B4B; padding: 2rem; color: white;">
        Test Color: #FF4B4B
    </div>
""", unsafe_allow_html=True)
```

## Resources

- Colour Picker: <https://htmlcolorcodes.com/>
- Gradient Generator: <https://cssgradient.io/>
- Streamlit Theming Docs: <https://docs.streamlit.io/library/advanced-features/theming>
- Accessibility Checker: <https://webaim.org/resources/contrastchecker/>

## Current Implementation

Your app now includes:

- ✅ Custom dark theme via `config.toml`
- ✅ CSS injection for gradients and styling
- ✅ Colour-coded fraud/legit badges
- ✅ Custom button styling with hover effects
- ✅ Enhanced metric cards
- ✅ Gradient backgrounds for headers
