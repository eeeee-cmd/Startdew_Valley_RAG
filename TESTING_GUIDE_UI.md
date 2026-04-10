# Integration Testing Guide (Browser UI)

**Access the UI:** http://localhost:8000

**Server must be running:** `cd src2 && python -m uvicorn app:app --port 8000`

All tests below use the **Browser UI**. Simply type in the message box and hit Send/Enter.

The UI will:
- ✅ Auto-generate a `session_id` on first message
- ✅ Reuse the same session for all messages in that conversation
- ✅ Display action questions with guided prompts and suggestions
- ✅ Show validation errors with helpful guidance
- ✅ Display the final action result when complete

---

## Test 1: Basic Query (Session Auto-Generation)

**Purpose:** Verify sessions are auto-generated and normal RAG queries work

### Browser Steps:
1. Go to **http://localhost:8000**
2. Type: `How do I plant crops in spring?`
3. Hit Send

### Expected Response:
- ✅ Answer appears with farming advice
- ✅ Sources appear below with wiki references
- ✅ Session badge or ID visible (auto-generated)
- ✅ Intent shows as `CROPS`

---

## Test 2: Action Detection (Friendship Plan)

**Purpose:** Verify action intent detection initiates multi-turn flow

### Browser Steps:
1. Go to **http://localhost:8000** (fresh conversation)
2. Type: `Can you help me create a friendship plan to marry Abigail?`
3. Hit Send

### Expected Response:
- ✅ System recognizes action intent
- ✅ Guided question appears: **"Which villager do you want to romance?"**
- ✅ Shows list of 11 villagers to choose from
- ✅ **"Action in progress"** indicator appears
- ✅ Session ID is generated and visible

**Keep this tab open or note the URL for Test 3**

---

## Test 3: Multi-Turn Friendship Plan (Full Flow - 3 Parameters)

**Purpose:** Complete a full multi-turn action with parameter collection

### Turn 1: Provide Villager
1. **Type:** `Abigail`
2. **Hit Send**

**Expected:**
- ✅ Success message: "✅ Great! Abigail it is!"
- ✅ Next question appears: **"What's your current friendship level with Abigail?"**
- ✅ Shows range guide (0/4/8/10 hearts with meanings)
- ✅ Still shows **"Action in progress"**

### Turn 2: Provide Current Hearts
1. **Type:** `2`
2. **Hit Send**

**Expected:**
- ✅ Success message: "✅ 2 hearts with Abigail..."
- ✅ Next question appears: **"How many gifts can you give Abigail per week?"**
- ✅ Shows frequency options (1/3/5/7 gifts with meanings)
- ✅ Still shows **"Action in progress"**

### Turn 3: Provide Gifts Per Week (Action Executes)
1. **Type:** `3`
2. **Hit Send**

**Expected:**
- ✅ Success message: "✅ Perfect! 3 gifts per week..."
- ✅ **"Action in progress"** changes to `false`
- ✅ Action result displays with friendship plan details
- ✅ Shows estimated timeline to romance Abigail

---

## Test 4: Farm Plan Action (Multi-Turn - 2 Parameters)

**Purpose:** Test another multi-turn action with fewer parameters

### Turn 1: Start Farm Plan
1. **Fresh conversation** - Go to http://localhost:8000
2. **Type:** `Help me create a farm plan`
3. **Hit Send**

**Expected:**
- ✅ System recognizes farm action
- ✅ Question appears: **"How many crop plots do you have available?"**
- ✅ Shows examples (5-10 small, 15-25 medium, 50+ large)
- ✅ **"Action in progress"** indicator shows

### Turn 2: Provide Plot Count
1. **Type:** `20`
2. **Hit Send**

**Expected:**
- ✅ Success: "✅ 20 plots noted..."
- ✅ Next question: **"What's your budget for seeds for 20 plots?"**
- ✅ Shows budget tier examples (1000g/5000g/10000g+)
- ✅ Still **"Action in progress"**

### Turn 3: Provide Budget (Action Executes)
1. **Type:** `5000`
2. **Hit Send**

**Expected:**
- ✅ Success: "✅ 5000g budget set..."
- ✅ **"Action in progress"** becomes `false`
- ✅ Farm plan displays with crop recommendations
- ✅ Shows profitability calculations

---

## Test 5: Save Favorites (Single-Turn Action)

**Purpose:** Test single-turn action (no parameter collection)

### Browser Steps:
1. **Fresh conversation** - http://localhost:8000
2. **Type:** `Save my favorite gifts for Abigail and Sebastian`
3. **Hit Send**

**Expected:**
- ✅ Action completes immediately
- ✅ **"Action in progress"** is `false`
- ✅ Confirmation message: "Favorites saved!"
- ✅ Shows saved villager data

---

## Test 6: Conversation Memory (Session Reuse)

**Purpose:** Verify conversation history is tracked within a session

### Turn 1: First Query
1. **Fresh conversation** - http://localhost:8000
2. **Type:** `What crops are best for making money?`
3. **Hit Send**
4. **Note the answer** (e.g., "Melons, Starfruit, etc.")

### Turn 2: Follow-up Question
1. **Type:** `How long does it take to grow?`
2. **Hit Send**

**Expected:**
- ✅ System remembers the first question
- ✅ Answer references the crops mentioned before
- ✅ **Same session ID** used (visible in UI or browser dev tools)
- ✅ Shows context awareness

---

## Test 7: Off-Topic Query

**Purpose:** Verify off-topic rejection still works

### Browser Steps:
1. **Fresh conversation** - http://localhost:8000
2. **Type:** `What is the weather today?`
3. **Hit Send**

**Expected:**
- ✅ Response: "I'm designed to answer questions about Stardew Valley..."
- ✅ Off-topic rejection message appears
- ✅ **"Action in progress"** is `false`
- ✅ Suggests asking about farming, villagers, items, etc.

---

## Test 8: Invalid Action Parameter Handling

**Purpose:** Verify invalid parameters are rejected gracefully with guidance

### Start a Friendship Plan:
1. **Fresh conversation** - http://localhost:8000
2. **Type:** `Help me romance Abigail`
3. **Hit Send**
4. **Type:** `Abigail`
5. **Hit Send**

### Provide Invalid Heart Value (>10):
1. **Type:** `15`
2. **Hit Send**

**Expected:**
- ✅ Error message appears: "❌ Invalid heart level: 15"
- ✅ Shows guidance: "**Hearts must be 0-10**"
- ✅ Provides examples: "Try: 0, 2, 4, 6, 8, or 10"
- ✅ **"Action in progress"** stays `true`
- ✅ System re-asks for the same parameter

### Provide Valid Value:
1. **Type:** `5`
2. **Hit Send**

**Expected:**
- ✅ Success: "✅ 5 hearts with Abigail..."
- ✅ Continues to next parameter

---

## Test 9: Invalid Villager Name

**Purpose:** Verify villager name validation

### Browser Steps:
1. **Fresh conversation** - http://localhost:8000
2. **Type:** `Create a romance plan`
3. **Hit Send** (gets action detection)
4. **Type:** `InvalidName`
5. **Hit Send**

**Expected:**
- ✅ Error message: "❌ I don't recognize 'InvalidName'"
- ✅ Shows list of valid villagers
- ✅ Suggests choosing from the list
- ✅ System re-asks for villager name

### Provide Valid Name:
1. **Type:** `Sebastian`
2. **Hit Send**

**Expected:**
- ✅ Success: "✅ Great! Sebastian it is!"
- ✅ Continues to next parameter

---

## Test 10: Unknown Intent

**Purpose:** Verify ambiguous queries are handled

### Browser Steps:
1. **Fresh conversation** - http://localhost:8000
2. **Type:** `Tell me about Stardew Valley`
3. **Hit Send**

**Expected:**
- ✅ Intent shows as `UNKNOWN`
- ✅ **"Action in progress"** is `false`
- ✅ General Stardew information is provided
- ✅ Sources appear with wiki references

---

## UI Testing Checklist

**Session & Basic:**
- [ ] Test 1: Fresh query auto-generates session
- [ ] Test 6: Session memory works (follow-up question understood)

**Actions - Multi-Turn:**
- [ ] Test 2: Friendship plan action detected
- [ ] Test 3: Full friendship plan (3 turns) completes with result
- [ ] Test 4: Farm plan action (2 turns) completes with result
- [ ] Test 5: Save favorites (single-turn) completes immediately

**Guided Prompts & Error Handling:**
- [ ] Test 8: Invalid parameters show helpful error + examples
- [ ] Test 9: Invalid villager name shows list of valid villagers
- [ ] Test 2/3/4: Prompts show suggestions (villager list, heart examples, etc.)

**Safety & Intent Routing:**
- [ ] Test 7: Off-topic queries rejected appropriately
- [ ] Test 10: Unknown intents routed to DefaultAgent

---

## What to Look For in the UI

✅ **Guided Prompts Should Show:**
- Clear question text (bold/emphasized)
- Bullet point suggestions with examples
- Range information (0-10, 1-7, etc.)
- Meaning of values (0 hearts = Just met, etc.)

✅ **Error Messages Should Show:**
- ❌ What was invalid
- 📌 What the valid range is
- 💡 Specific examples to try
- ↩️ System re-asks for the parameter

✅ **Action Results Should Show:**
- ✅ Completion confirmation
- 📋 Plan details (friendship timeline, farm recommendations, etc.)
- 💾 Save confirmation (if applicable)
- 🎯 Actionable next steps

✅ **Session Indicators:**
- Session ID visible (in URL, badge, or message)
- Same session for all turns in one conversation
- New session for fresh conversation (new tab/browser)

---

## All Tests Passing?

**✅ Ready to commit!**

```bash
cd /Users/lamanamulaffer/Documents/GitHub/Startdew_Valley_RAG
git add TESTING_GUIDE_UI.md
git commit -m "Add browser UI testing guide with step-by-step instructions"
git push origin main
```
