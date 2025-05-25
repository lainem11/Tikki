package com.example.tikki;


import androidx.annotation.DrawableRes;
import androidx.appcompat.app.AppCompatActivity;

import android.animation.Animator;
import android.animation.AnimatorInflater;
import android.animation.AnimatorListenerAdapter;
import android.animation.AnimatorSet;
import android.animation.ObjectAnimator;
import android.animation.PropertyValuesHolder;
import android.annotation.SuppressLint;
import android.app.Dialog;
import android.content.ClipData;
import android.content.ClipDescription;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.TransitionDrawable;
import android.media.SoundPool;
import android.os.Bundle;

import org.pytorch.Module;
import org.pytorch.LiteModuleLoader;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import android.os.Handler;
import android.os.Looper;
import android.text.Html;
import android.view.DragEvent;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.view.animation.AlphaAnimation;
import android.view.animation.AnticipateOvershootInterpolator;
import android.view.animation.DecelerateInterpolator;
import android.view.animation.OvershootInterpolator;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.util.Log;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;

import java.util.*;
import java.util.ArrayList;
import java.util.List;

import android.graphics.drawable.AnimationDrawable;

import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.text.HtmlCompat;
import androidx.dynamicanimation.animation.DynamicAnimation;
import androidx.dynamicanimation.animation.SpringAnimation;
import androidx.dynamicanimation.animation.SpringForce;
import androidx.interpolator.view.animation.FastOutSlowInInterpolator;

public class GameActivity2p extends AppCompatActivity {
    private Module module = null;

    private Tikki game = null;
    // Given the name of the pytorch model, get the path for that model
    public String assetFilePath(String assetName) throws IOException {
        File file = new File(this.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = this.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
    private ConstraintLayout rootLayout;
    private LinearLayout pileLayout0,pileLayout1;
    private FrameLayout handLayout0,handLayout1;
    private View pulseOverlay0, pulseOverlay1;
    private Animation scaleAlphaAnimation;
    private AnimationDrawable pulseAnimation;
    private HashMap<String, Integer> cardImageMap;

    private ImageButton newMatchButton;
    private ImageButton passButton,winButton,twoWinButton,quitButton;
    private Cards selectedCard = null;
    private String selectedBet = null;
    private List<ImageView> player0Cards = new ArrayList<>();
    private List<ImageView> player1Cards = new ArrayList<>();
    private TextView score0,score1,text_player0Name,text_player1Name,player0_speech,player1_speech,text_game_end,total_score,damageText0,damageText1;
    private Object playedMove = null;
    private int humanPlayerIndex = 0;
    private int roundStarted = 0;
    private Handler handler;
    private String backside = "backside3";
    private boolean isBotPlaying = false; // Track if bot logic is active
    private final Random random = new Random(); // Shared random instance

    private double cardOffset = 0.7;
    private int gamesPlayed = 0;
    private int gamesWon = 0;
    private List<ImageView> cardsInTable = new ArrayList<>();
    private float velocityX = 0, velocityY = 0;
    private float lastX, lastY;
    private long lastTime;
    private int draggedCardIndex;
    private SoundPool soundPool;
    private int cardFlapSoundId1,cardFlapSoundId2,cardFlapSoundId3,cardSuhSoundId,clingitySoundId,loseSoundId, winSoundId,nipetiSoundId,snapSoundId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.tikki_main_screen_2p);

        rootLayout = findViewById(R.id.root_layout);
        Intent intent = getIntent();
        String player0Name = intent.getStringExtra("playerName");
        String enemyModel = intent.getStringExtra("enemyModel");

        cardImageMap = new HashMap<>();
        cardImageMap.put("♥2",R.drawable.two_of_hearts);
        cardImageMap.put("♥3",R.drawable.three_of_hearts);
        cardImageMap.put("♥4",R.drawable.four_of_hearts);
        cardImageMap.put("♥5",R.drawable.five_of_hearts);
        cardImageMap.put("♥6",R.drawable.six_of_hearts);
        cardImageMap.put("♥7",R.drawable.seven_of_hearts);
        cardImageMap.put("♥8",R.drawable.eight_of_hearts);
        cardImageMap.put("♥9",R.drawable.nine_of_hearts);
        cardImageMap.put("♥10",R.drawable.ten_of_hearts);
        cardImageMap.put("♥11",R.drawable.jack_of_hearts);
        cardImageMap.put("♥12",R.drawable.queen_of_hearts);
        cardImageMap.put("♥13",R.drawable.king_of_hearts);
        cardImageMap.put("♥14",R.drawable.ace_of_hearts);

        cardImageMap.put("♦2",R.drawable.two_of_diamonds);
        cardImageMap.put("♦3",R.drawable.three_of_diamonds);
        cardImageMap.put("♦4",R.drawable.four_of_diamonds);
        cardImageMap.put("♦5",R.drawable.five_of_diamonds);
        cardImageMap.put("♦6",R.drawable.six_of_diamonds);
        cardImageMap.put("♦7",R.drawable.seven_of_diamonds);
        cardImageMap.put("♦8",R.drawable.eight_of_diamonds);
        cardImageMap.put("♦9",R.drawable.nine_of_diamonds);
        cardImageMap.put("♦10",R.drawable.ten_of_diamonds);
        cardImageMap.put("♦11",R.drawable.jack_of_diamonds);
        cardImageMap.put("♦12",R.drawable.queen_of_diamonds);
        cardImageMap.put("♦13",R.drawable.king_of_diamonds);
        cardImageMap.put("♦14",R.drawable.ace_of_diamonds);

        cardImageMap.put("♣2",R.drawable.two_of_clubs);
        cardImageMap.put("♣3",R.drawable.three_of_clubs);
        cardImageMap.put("♣4",R.drawable.four_of_clubs);
        cardImageMap.put("♣5",R.drawable.five_of_clubs);
        cardImageMap.put("♣6",R.drawable.six_of_clubs);
        cardImageMap.put("♣7",R.drawable.seven_of_clubs);
        cardImageMap.put("♣8",R.drawable.eight_of_clubs);
        cardImageMap.put("♣9",R.drawable.nine_of_clubs);
        cardImageMap.put("♣10",R.drawable.ten_of_clubs);
        cardImageMap.put("♣11",R.drawable.jack_of_clubs);
        cardImageMap.put("♣12",R.drawable.queen_of_clubs);
        cardImageMap.put("♣13",R.drawable.king_of_clubs);
        cardImageMap.put("♣14",R.drawable.ace_of_clubs);

        cardImageMap.put("♠2",R.drawable.two_of_spades);
        cardImageMap.put("♠3",R.drawable.three_of_spades);
        cardImageMap.put("♠4",R.drawable.four_of_spades);
        cardImageMap.put("♠5",R.drawable.five_of_spades);
        cardImageMap.put("♠6",R.drawable.six_of_spades);
        cardImageMap.put("♠7",R.drawable.seven_of_spades);
        cardImageMap.put("♠8",R.drawable.eight_of_spades);
        cardImageMap.put("♠9",R.drawable.nine_of_spades);
        cardImageMap.put("♠10",R.drawable.ten_of_spades);
        cardImageMap.put("♠11",R.drawable.jack_of_spades);
        cardImageMap.put("♠12",R.drawable.queen_of_spades);
        cardImageMap.put("♠13",R.drawable.king_of_spades);
        cardImageMap.put("♠14",R.drawable.ace_of_spades);
        cardImageMap.put("backside3", R.drawable.backside3);

        try {
            module = LiteModuleLoader.load(assetFilePath(enemyModel));
        } catch (IOException e) {
            Log.e("PTRTDryRun","Error reading assets",e);
            finish();
        }
        handLayout0 = findViewById(R.id.cards_container0);
        handLayout1 = findViewById(R.id.cards_container1);
        pulseOverlay0 = findViewById(R.id.pulseOverlay0);
        pulseOverlay1 = findViewById(R.id.pulseOverlay1);
        pileLayout0 = findViewById(R.id.pileLayout0);
        pileLayout1 = findViewById(R.id.pileLayout1);

        score0 = findViewById(R.id.player0_score);
        score1 = findViewById(R.id.player1_score);
        total_score = findViewById(R.id.total_score);
        text_player0Name = findViewById(R.id.text_player0Name);
        text_player1Name = findViewById(R.id.text_player1Name);
        player0_speech = findViewById(R.id.player0_speech);
        player1_speech = findViewById(R.id.player1_speech);
        damageText0 = findViewById(R.id.damage_text0);
        damageText1 = findViewById(R.id.damage_text1);

        newMatchButton = findViewById(R.id.new_match_button);
        passButton = findViewById(R.id.pass_button);
        winButton = findViewById(R.id.win_button);
        twoWinButton = findViewById(R.id.two_win_button);
        quitButton = findViewById(R.id.buttonQuit);
        text_game_end = findViewById(R.id.text_game_end);

        text_game_end.setVisibility(View.GONE);
        pulseOverlay0.setVisibility(View.GONE);
        pulseOverlay1.setVisibility(View.GONE);
        passButton.setVisibility(View.GONE);
        winButton.setVisibility(View.GONE);
        twoWinButton.setVisibility(View.GONE);
        damageText0.setVisibility(View.GONE);
        damageText1.setVisibility(View.GONE);


        pulseAnimation = (AnimationDrawable) pulseOverlay0.getBackground();
        pulseAnimation.setEnterFadeDuration(3000);
        pulseAnimation.setExitFadeDuration(3000);

        scaleAlphaAnimation = AnimationUtils.loadAnimation(this, R.anim.pulse);

        // Initialize SoundPool
        soundPool = new SoundPool.Builder()
                .setMaxStreams(5) // Allow up to 5 simultaneous sounds
                .build();

        // Load sounds
        cardFlapSoundId1 = soundPool.load(this, R.raw.card_flap1, 1);
        cardFlapSoundId2 = soundPool.load(this, R.raw.card_flap2, 1);
        cardFlapSoundId3 = soundPool.load(this, R.raw.card_flap3, 1);
        cardSuhSoundId = soundPool.load(this, R.raw.card_suh, 1);
        clingitySoundId = soundPool.load(this, R.raw.clingity, 1);
        loseSoundId = soundPool.load(this, R.raw.lose_sound, 1);
        winSoundId = soundPool.load(this, R.raw.win_sound, 1);
        nipetiSoundId = soundPool.load(this, R.raw.nipeti, 1);
        snapSoundId = soundPool.load(this,R.raw.snap,1);

        soundPool.play(snapSoundId, 1.0f, 1.0f, 1, 0, 1.0f);

        // Get parent container
        ViewGroup betContainer = findViewById(R.id.bet_container);
        setExpandingCircleEffect(passButton,betContainer);
        setExpandingCircleEffect(winButton,betContainer);
        setExpandingCircleEffect(twoWinButton,betContainer);

        setTotalScoreTExt(0, 0);

        // Start tikki game
        String player1Name = "";
        if (enemyModel.equals("mobile_model_final_hard_test2.pt")) {
            player1Name = "Brainy";
        }
        if (enemyModel.equals("mobile_model_easy.pt")) {
            player1Name = "Dummy";
        }
        text_player0Name.setText(getString(R.string.player_name,player0Name));
        text_player1Name.setText(getString(R.string.player_name,player1Name));
        Player P1 = new Player(player0Name,null);
        Player P2 = new Player(player1Name,module);
        List<Player> players = new ArrayList<Player>();
        players.add(P1);
        players.add(P2);
        handler = new Handler(Looper.getMainLooper());
        game = startGame(players);

        /* DEBUGGING
        try {
            module = LiteModuleLoader.load(assetFilePath("mobile_model_final_hard_test2.pt"));
            long[] long_state = new long[]{47, 6, 21, 37, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 61};
            Tensor state = Tensor.fromBlob(long_state, new long[]{1, 19});
            Tensor output = module.forward(IValue.from(state)).toTensor();
            Log.d("SimpleModelOutput", Arrays.toString(output.getDataAsFloatArray()));
        } catch (IOException e) {
            Log.e("PTRTDryRun", "Error reading assets", e);
            finish();
        }

        game.players.get(1).hand.set(0, new Cards("♣", "13"));
        game.players.get(1).hand.set(1, new Cards("♦", "3"));
        game.players.get(1).hand.set(2, new Cards("♥", "7"));
        game.players.get(1).hand.set(3, new Cards("♥", "11"));
        game.players.get(1).hand.set(4, new Cards("♦", "9"));

        int[] state = game.players.get(1).getState(game);
        float [] action_array = game.players.get(1).chooseAction(game);
        String moveString = game.players.get(1).playBet(game);
        */

        // Get the ImageButton by ID
        ImageButton imageButtonRules = findViewById(R.id.imageButtonRules);

        // Set a click listener
        imageButtonRules.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Create and display the dialog
                showHelpDialog();
            }
        });

        // Set card drag-and-drop listeners
        setCardDropCallbacks();


        passButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!canUserInteract()) return; // Early exit if bot is playing
                setButtonScaleAnimation(v);
                selectedBet = "pass";
                fadeDrawable(passButton, R.drawable.pass_button_enabled,100);
                winButton.setImageResource(R.drawable.win_button_disabled);
                twoWinButton.setImageResource(R.drawable.two_win_button_disabled);
                takeGameStep();
            }
        });

        winButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!canUserInteract()) return; // Early exit if bot is playing
                setButtonScaleAnimation(v);
                selectedBet = "win";
                passButton.setImageResource(R.drawable.pass_button_disabled);
                fadeDrawable(winButton, R.drawable.win_button_enabled,100);
                twoWinButton.setImageResource(R.drawable.two_win_button_disabled);
                takeGameStep();
            }
        });

        twoWinButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!canUserInteract()) return; // Early exit if bot is playing
                setButtonScaleAnimation(v);
                selectedBet = "2-win";
                passButton.setImageResource(R.drawable.pass_button_disabled);
                winButton.setImageResource(R.drawable.win_button_disabled);
                fadeDrawable(twoWinButton, R.drawable.two_win_button_enabled,100);
                takeGameStep();
            }
        });

        newMatchButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                if (!canUserInteract()) return; // Early exit if bot is playing
                setButtonScaleAnimation(v);
                //newMatchButton.setVisibility(View.INVISIBLE);
                if (game.gameWinner != null) {
                    game = startGame(players);
                } else {
                    start_round();
                    play_bot_moves();
                }
            }
        });

        quitButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Intent intent = new Intent(GameActivity2p.this, MainActivity.class);
                startActivity(intent);
            }
        });
    }
    @Override
    protected void onPause() {
        super.onPause();
        soundPool.autoPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        soundPool.autoResume();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        soundPool.release();
        soundPool = null;
    }

    private void setTotalScoreTExt(int wins, int totalGames) {
        String text = "<font color=#f2eae7>Wins\n</font> <font color=#456fd8>" + wins + "</font>" +  "<font color=#f2eae7> / </font>" + "<font color=#456fd8>" + totalGames + "</font>";
        total_score.setText(Html.fromHtml(text,Html.FROM_HTML_MODE_LEGACY));
    }
    // Function to apply expanding circle effect
    private void setExpandingCircleEffect(View button, ViewGroup parent) {
        button.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_DOWN) {
                createExpandingCircle(this, parent, event.getX(), event.getY(), v);
            }
            return false; // Ensures button click still works
        });
    }

    // Method to create an expanding circle effect
    private void createExpandingCircle(Context context, ViewGroup parent, float x, float y, View targetView) {
        final View rippleView = new View(context);
        rippleView.setLayoutParams(new ViewGroup.LayoutParams(100, 100));
        rippleView.setBackground(createCircleDrawable());

        // Position the ripple at the touch location
        rippleView.setX(targetView.getX() + x - 50);
        rippleView.setY(targetView.getY() + y - 50);
        rippleView.setTranslationZ(-1f);
        rippleView.setClipToOutline(false);
        parent.addView(rippleView);

        // Animate expansion
        ObjectAnimator scaleX = ObjectAnimator.ofFloat(rippleView, View.SCALE_X, 1f, 5f);
        ObjectAnimator scaleY = ObjectAnimator.ofFloat(rippleView, View.SCALE_Y, 1f, 5f);
        ObjectAnimator fadeOut = ObjectAnimator.ofFloat(rippleView, View.ALPHA, 0.25f, 0f);

        scaleX.setInterpolator(new FastOutSlowInInterpolator());
        scaleY.setInterpolator(new FastOutSlowInInterpolator());
        fadeOut.setInterpolator(new DecelerateInterpolator());

        scaleX.setDuration(1000);
        scaleY.setDuration(1000);
        fadeOut.setDuration(1000);

        scaleX.start();
        scaleY.start();
        fadeOut.start();

        // Remove view after animation ends
        fadeOut.addListener(new AnimatorListenerAdapter() {
            @Override
            public void onAnimationEnd(Animator animation) {
                parent.removeView(rippleView);
            }
        });
    }

    // Create a circular drawable for the ripple
    private GradientDrawable createCircleDrawable() {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setShape(GradientDrawable.OVAL);
        drawable.setColor(Color.parseColor("#66FFFFFF")); // Transparent white
        return drawable;
    }

    private void setButtonScaleAnimation(View v) {
        // Scale animation
        ObjectAnimator scaleAnim = ObjectAnimator.ofPropertyValuesHolder(
                v,
                PropertyValuesHolder.ofFloat(View.SCALE_X, 1.0f, 0.9f, 1.0f),
                PropertyValuesHolder.ofFloat(View.SCALE_Y, 1.0f, 0.9f, 1.0f)
        );
        scaleAnim.setDuration(250);
        scaleAnim.setInterpolator(new DecelerateInterpolator());

        // Alpha animation (fade out)
        ObjectAnimator alphaAnim = ObjectAnimator.ofFloat(v, View.ALPHA, 1.0f, 0.0f);
        alphaAnim.setDuration(250);
        alphaAnim.setInterpolator(new DecelerateInterpolator());

        // Combine animations to run concurrently
        AnimatorSet animatorSet = new AnimatorSet();
        animatorSet.playTogether(scaleAnim, alphaAnim);
        animatorSet.start();
    }
    private void showHelpDialog() {
        // Create a new Dialog
        Dialog dialog = new Dialog(this);
        dialog.setContentView(R.layout.help_menu); // Set the dialog layout

        // Set up the close button inside the dialog
        Button closeButton = dialog.findViewById(R.id.closeButton);
        closeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                dialog.dismiss(); // Close the dialog
            }
        });

        // Optionally set dynamic text for the help content
        TextView helpTextView = dialog.findViewById(R.id.helpTextView);
        dialog.getWindow().setLayout((int) (getResources().getDisplayMetrics().widthPixels * 0.9),  // 80% of screen width
                WindowManager.LayoutParams.WRAP_CONTENT); // Height wraps content
        helpTextView.setText(HtmlCompat.fromHtml(
                "Reach 5 points to win the game by winning matches. Each match has a betting round and five tricks.<br><br>" +
                        "Tricks: The first card sets the lead suit. Follow suit if possible; the highest lead suit card wins. The winner of the last trick wins the match.<br><br>" +
                        "<b>Scoring:</b><br>" +
                        "• Match winner: +1 point.<br>" +
                        "• If the winning card is a 2: +1 extra point.<br><br>" +
                        "<b>Betting:</b><br>" +
                        "• Pass: No change.<br>" +
                        "• Win: +1 if you win the match, else -1.<br>" +
                        "• 2-win: +2 if you win the match with a 2, else -2.",
                HtmlCompat.FROM_HTML_MODE_LEGACY
        ));
        // Show the dialog
        dialog.show();
    }

    private boolean canUserInteract() {
        if (isBotPlaying) {
            // Optional: Notify the user
            //Toast.makeText(this, "Not your turn!", Toast.LENGTH_SHORT).show();
            return false;
        }
        return true;
    }

    private  void takeGameStep() {
        lowerCards();
        advanceGame();
        if (game.gameOver) {
            endGame();
        }
    }

    public void start_round() {
        // Reset the round state and show all hands
        cardsInTable = new ArrayList<>();
        for (int i = 0; i < game.players.size(); i++) {
            showHand(game.players, i);
        }
        roundStarted = 1;
        updateTurnIndicator(game.currentPlayerIndex);
        resetSelection();

        if (game.currentPlayerIndex == humanPlayerIndex) {
            isBotPlaying = false;
        } else {
            isBotPlaying = true;
        }
    }

    public void applyDamageNumberAnimation(TextView textView, int damage) {
        String speechString = "";
        if (damage > 0) {
            speechString = "+" + String.valueOf(damage); // Prepend "+"
            textView.setTextColor(getResources().getColor(R.color.point_add_color, null));
        } else {
            speechString = String.valueOf(damage); // Prepend "-"
            textView.setTextColor(getResources().getColor(R.color.point_reduce_color, null));
        }

        // Set the text in the TextView
        textView.setText(speechString);
        textView.setVisibility(View.VISIBLE);

        // Reset initial state
        textView.setAlpha(0f); // Start fully transparent
        textView.setTranslationX(0);
        textView.setTranslationY(0);

        // Random X translation
        Random random = new Random();
        double randomXTranslation = ((random.nextDouble() - 0.5) * handLayout0.getWidth() / 2);

        // Animation parameters
        float direction = -1;
        long fadeInDuration = 800;
        long fadeOutDuration = 800;
        long fadeOutStartOffset = 800;
        long translationDuration = fadeInDuration + fadeOutStartOffset + fadeOutDuration; // 800 + 800 + 800 = 2400ms

        // Translation animation (runs concurrently for the entire duration)
        textView.animate()
                .translationYBy((float) (getApplicationContext().getResources().getDisplayMetrics().heightPixels * 0.02 * direction)) // Move vertically
                .translationXBy((float) (randomXTranslation * 0.25)) // Random horizontal movement
                .setDuration(translationDuration) // Spans fade-in, pause, and fade-out
                .setInterpolator(new DecelerateInterpolator()) // Apply non-linear motion
                .start();

        // Fade-in animation
        textView.animate()
                .alpha(1.0f) // Fade in
                .setDuration(fadeInDuration) // 800ms
                .withEndAction(() -> {
                    // Start fade-out animation
                    textView.animate()
                            .alpha(0.0f) // Fade out
                            .setDuration(fadeOutDuration) // 800ms
                            .setStartDelay(fadeOutStartOffset) // 800ms pause
                            .withEndAction(() -> {
                                // Reset position and alpha for repeatable animation
                                textView.setTranslationX(0);
                                textView.setTranslationY(0);
                                textView.setAlpha(0.0f);
                            })
                            .start();
                })
                .start();

        soundPool.play(clingitySoundId, 0.7f, 0.7f, 1, 0, 1.0f);
    }

    public void applySpeechTextAnimation(TextView textView, String moveString, int playerIndex) {
        String speechString = "";

        // Determine the speech text based on moveString
        if (moveString.equals("pass")) {
            speechString = "Pass";
        } else if (moveString.equals("win")) {
            speechString = "Win";
        } else if (moveString.equals("2-win")) {
            speechString = "2-win";
        }

        // Set the text in the TextView
        textView.setText(speechString);

        // Fade-in animation
        final Animation fadeIn = new AlphaAnimation(0.0f, 1.0f);
        fadeIn.setDuration(500);
        fadeIn.setFillAfter(true); // Ensure the fadeIn effect persists

        // Fade-out animation
        final Animation fadeOut = new AlphaAnimation(1.0f, 0.0f);
        fadeOut.setDuration(500);
        fadeOut.setStartOffset(500);
        fadeOut.setFillAfter(true); // Ensure the fadeOut effect persists

        // Movement animation (random X translation)
        Random random = new Random();
        double randomXTranslation = ((random.nextDouble()-0.5) * handLayout0.getWidth() / 2);

        // Start movement animation
        float direction = 1;
        if (playerIndex == 0) {
            direction = -1;
        }
        textView.animate()
                .translationYBy((float) (getApplicationContext().getResources().getDisplayMetrics().heightPixels * 0.15 * direction)) // Move vertically
                .translationXBy((float) randomXTranslation) // Random horizontal movement
                .setDuration(fadeOut.getDuration() + fadeOut.getStartOffset() + 1000)
                .setInterpolator(new DecelerateInterpolator()) // Apply non-linear motion
                .withEndAction(() -> {
                    // Reset position for repeatable animation
                    textView.setTranslationX(0);
                    textView.setTranslationY(0);
                })
                .start();

        soundPool.play(cardSuhSoundId, 0.5f, 0.5f, 1, 0, 1.0f);

        // Set up fade-in listener to start fade-out after fade-in ends
        fadeIn.setAnimationListener(new Animation.AnimationListener() {
            @Override
            public void onAnimationStart(Animation animation) {
                // No action needed here
            }

            @Override
            public void onAnimationRepeat(Animation animation) {
                // No action needed here
            }

            @Override
            public void onAnimationEnd(Animation animation) {
                // Start fade-out when fade-in ends
                textView.startAnimation(fadeOut);
            }
        });

        // Start the fade-in animation
        textView.startAnimation(fadeIn);
    }
    private void play_bot_moves() {
        if (game.currentPlayerIndex != humanPlayerIndex) {
            isBotPlaying = true;
            // Generate a random delay using normal distribution
            int randomDelay = getNormalDelay(750, 300);
            // Delay the bot's move to simulate thinking
            handler.postDelayed(() -> {
                    advanceGame();
                    isBotPlaying = false;
            }, randomDelay);
        } else {
            player_turn_setup();
        }
    }

    private void player_turn_setup() {

        if (game.roundNumber == 0) {
            showBetButtons();
        } else {
            raisePlayableCards();
        }
    }

    private int getNormalDelay(int mean, int stdDev) {
        double gaussian = random.nextGaussian(); // Generate a random value from a standard normal distribution
        int delay = (int) (mean + gaussian * stdDev); // Scale and shift

        // Clamp delay to a valid range (e.g., 100ms to 1000ms)
        return Math.max(500, Math.min(2000, delay));
    }

    private void advanceGame() {
        Log.d("Logic debug","Advancing game");
        String moveString = "";

        if (!game.gameOver) {
            // Proceed with the next turn if game is not over
            int cardPlayer = game.currentPlayerIndex;
            int oldRoundNumber = game.roundNumber;

            // Check if move is based on selection
            if (selectedBet != null) {
                playedMove = selectedBet;
                hideBetButtons();
                resetSelection();
            } else if (selectedCard != null) {
                playedMove = selectedCard;
                resetSelection();
            } else {
                playedMove = null;
            }
            Boolean hasBrainBool = game.players.get(cardPlayer).brains == null;
            int hasBrain = hasBrainBool ? 1 : 0;
            Boolean hasSelection = playedMove == null;
            int selectionFilled = hasSelection ? 0 : 1;

            // Play turn and update visual elements
            Object[] result = game.nextTurn(playedMove);
            moveString = (String) result[0];

            if (oldRoundNumber == 0) {
                TextView player_speech = cardPlayer == 0 ? player0_speech : player1_speech;
                applySpeechTextAnimation(player_speech, moveString, cardPlayer);
            }

            updateTurnIndicator(game.currentPlayerIndex);

            // If card was played (not round 0), show card play animation
            if (oldRoundNumber != 0) {
                animateCardPlay(cardPlayer, moveString);
                roundStarted = 0;
            }

            // Start next round if round ended on this turn
            if (game.roundNumber == 0 && roundStarted == 0) {
                setScores();
                soundPool.play(clingitySoundId, 1.0f, 1.0f, 1, 0, 1.0f);
                hideTurnIndicator();
                //fadeDrawable(nextTurnButton, R.drawable.next_button_enabled,100);
                newMatchButton.setVisibility(View.VISIBLE);
                ObjectAnimator fadein = ObjectAnimator.ofFloat(newMatchButton, "alpha", 0f, 1f);
                fadein.setDuration(500); // Fade-in duration
                fadein.start();
            } else {
                play_bot_moves();
            }

            // Go the game end if game over
            if (game.gameOver) {
                setScores();
                endGame();
            }
        }


    }

    private void hideTurnIndicator() {
        pulseOverlay0.clearAnimation();
        pulseOverlay1.clearAnimation();
        pulseOverlay0.setVisibility(View.GONE);
        pulseOverlay1.setVisibility(View.GONE);
    }

    private void endGame() {
        resetSelection();
        hideTurnIndicator();
        text_game_end.setText(getString(R.string.winner_name, game.gameWinner.name));

        // Set the winner's name text
        text_game_end.setText(getString(R.string.winner_name, game.gameWinner.name));
        text_game_end.setGravity(Gravity.CENTER);
        Animation winnerAnimation = AnimationUtils.loadAnimation(this, R.anim.winner_announcement);
        text_game_end.startAnimation(winnerAnimation);

        text_game_end.setVisibility(View.VISIBLE);
        gamesPlayed += 1;
        if (game.gameWinner == game.players.get(0)) {
            soundPool.play(winSoundId, 1.0f, 1.0f, 1, 0, 1.0f);
            gamesWon += 1;
        } else {
            soundPool.play(loseSoundId, 1.0f, 1.0f, 1, 0, 1.0f);
        }
        setTotalScoreTExt(gamesWon, gamesPlayed);

        //fadeDrawable(nextTurnButton, R.drawable.next_button_enabled,100);
        newMatchButton.setVisibility(View.VISIBLE);
    }

    @SuppressLint("SetTextI18n")
    private void setScores() {
        // if lastKnownPlayerScores
        Map<Player, Integer> playerScores = game.playerScores;
        List<Player> players = game.players;
        String player0_score_string = Objects.requireNonNull(playerScores.get(players.get(0))).toString();
        String player1_score_string = Objects.requireNonNull(playerScores.get(players.get(1))).toString();

        CharSequence old_score0 = score0.getText();
        CharSequence old_score1 = score1.getText();
        score0.setText(player0_score_string);
        score1.setText(player1_score_string);

        // Animate if score changed
        Animation winnerAnimation = AnimationUtils.loadAnimation(this, R.anim.winner_announcement);
        if (!old_score0.equals(score0.getText())) {
            score0.startAnimation(winnerAnimation);
            int damage = subtractCharSequences(score0.getText(), old_score0);
            applyDamageNumberAnimation(damageText0, damage);
        }
        if (!old_score1.equals(score1.getText())) {
            score1.startAnimation(winnerAnimation);
            int damage = subtractCharSequences(score1.getText(), old_score1);
            applyDamageNumberAnimation(damageText1, damage);
        }



        //score0.setText(Objects.requireNonNull(playerScores.get(players.get(1))).toString());

        //text_game_end.setVisibility(View.VISIBLE);
    }
    public int subtractCharSequences(CharSequence cs1, CharSequence cs2) {
        try {
            int num1 = Integer.parseInt(cs1.toString());
            int num2 = Integer.parseInt(cs2.toString());
            return num1 - num2;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid number format in input: " + cs1 + " or " + cs2, e);
        }
    }
    private Tikki startGame(List<Player> players) {
        //sound.start();
        game = new Tikki(players);
        game.newGame();
        text_game_end.setVisibility(View.GONE);
        resetSelection();
        start_round();
        score0.setText("0");
        score1.setText("0");
        play_bot_moves();
        return game;
    }
    private void hideBetButtons() {
        passButton.setVisibility(View.GONE);
        winButton.setVisibility(View.GONE);
        twoWinButton.setVisibility(View.GONE);
    }

    private void showBetButtons() {
        passButton.setVisibility(View.VISIBLE);
        winButton.setVisibility(View.VISIBLE);
        twoWinButton.setVisibility(View.VISIBLE);
        ObjectAnimator fadein = ObjectAnimator.ofFloat(passButton, "alpha", 0f, 1f);
        fadein.setDuration(200); // Fade-in duration
        ObjectAnimator fadein2 = ObjectAnimator.ofFloat(winButton, "alpha", 0f, 1f);
        fadein.setDuration(200); // Fade-in duration
        ObjectAnimator fadein3 = ObjectAnimator.ofFloat(twoWinButton, "alpha", 0f, 1f);
        fadein.setDuration(200); // Fade-in duration
        fadein.start();
        fadein2.start();
        fadein3.start();
        soundPool.play(nipetiSoundId, 1.0f, 1.0f, 1, 0, 1.0f);
    }

    private boolean isPlayerTurn(int playerIndex) {
        return game.currentPlayerIndex == playerIndex;
    }

    private void resetSelection() {
        selectedCard = null;
        selectedBet = null;
        passButton.setImageResource(R.drawable.pass_button_disabled);
        winButton.setImageResource(R.drawable.win_button_disabled);
        twoWinButton.setImageResource(R.drawable.two_win_button_disabled);
    }

    private void fadeOutAndClearAnimation(View view) {
        // Create fade-out animation
        Animation fadeOut = new AlphaAnimation(1.0f, 0.0f);
        fadeOut.setDuration(500); // Duration of the fade-out effect (e.g., 500ms)
        fadeOut.setFillAfter(true); // Ensure the view stays invisible after fading out

        // Set an animation listener to clear animation after fade-out
        fadeOut.setAnimationListener(new Animation.AnimationListener() {
            @Override
            public void onAnimationStart(Animation animation) {}

            @Override
            public void onAnimationEnd(Animation animation) {
                // Clear the animation after fade-out completes
                view.clearAnimation();
            }

            @Override
            public void onAnimationRepeat(Animation animation) {}
        });

        // Start fade-out animation
        view.startAnimation(fadeOut);
    }

    private void fadeDrawable(ImageButton button, @DrawableRes int newDrawableRes, int duration) {
        // Get the current drawable and the new drawable
        Drawable currentDrawable = button.getDrawable();
        Drawable newDrawable = getResources().getDrawable(newDrawableRes, null);

        // Ensure currentDrawable is not null
        if (currentDrawable == null) {
            button.setImageDrawable(newDrawable);
            return;
        }

        // Create a TransitionDrawable with both drawables
        Drawable[] layers = new Drawable[]{currentDrawable, newDrawable};
        TransitionDrawable transitionDrawable = new TransitionDrawable(layers);
        transitionDrawable.setCrossFadeEnabled(true); // Enable proper blending

        // Set the TransitionDrawable as the new image
        button.setImageDrawable(transitionDrawable);

        // Start the transition
        transitionDrawable.startTransition(duration);
    }

    private void updateTurnIndicator(int currentPlayerIndex) {
        // Stop animations on both layouts
        if (pulseOverlay0.getVisibility() != View.GONE) {
            fadeOutAndClearAnimation(pulseOverlay0);
        }
        if (pulseOverlay1.getVisibility() != View.GONE) {
            fadeOutAndClearAnimation(pulseOverlay1);
        }

        // Start the animation on the current player's hand layout
        if (currentPlayerIndex == 0) {
            pulseOverlay0.setVisibility(View.VISIBLE);
            pulseOverlay0.startAnimation(scaleAlphaAnimation); // Start pulse animation on player 1 overlay
            pulseAnimation.start();
            pulseOverlay1.setVisibility(View.GONE);
        } else if (currentPlayerIndex == 1) {
            pulseOverlay1.setVisibility(View.VISIBLE);
            pulseOverlay1.startAnimation(scaleAlphaAnimation); // Start pulse animation on player 1 overlay
            pulseAnimation.start();
            pulseOverlay0.setVisibility(View.GONE);
        }
    }

    private int[] getArcPos_hand0(View layout, float angle) {
        int layoutWidth = layout.getWidth();
        int arc_radius = (int) Math.round(layoutWidth * 1.0);

        int[] location = new int[2];
        layout.getLocationOnScreen(location); // top left
        int[] arcOrigin = new int[2];
        arcOrigin[0] = location[0] + layoutWidth/2; // typecast to double??
        arcOrigin[1] = location[1] + arc_radius ;

        int[] pos = new int[2];
        pos[0] = (int) Math.round(Math.cos(Math.toRadians(angle)) * arc_radius + arcOrigin[0] - location[0]);
        pos[1] = (int) Math.round(-Math.sin(Math.toRadians(angle)) * arc_radius + arcOrigin[1] - location[1]);
        return pos;
    }
    private void flip_card_anim(final ImageView imageView, final int newImageResource) {
        // Load the animations
        AnimatorSet setOut = (AnimatorSet) AnimatorInflater.loadAnimator(this, R.animator.card_flip_out);
        AnimatorSet setIn = (AnimatorSet) AnimatorInflater.loadAnimator(this, R.animator.card_flip_in);

        // Set the target for the first animation
        setOut.setTarget(imageView);

        // Listener to switch the resource and start the second animation
        setOut.addListener(new AnimatorListenerAdapter() {
            @Override
            public void onAnimationStart(Animator animation) {
                // Start fade-out animation before flipping out
                ObjectAnimator fadein = ObjectAnimator.ofFloat(imageView, "alpha", 0f, 1f);
                fadein.setDuration(50); // Fade-in duration
                fadein.start();
            }
            @Override
            public void onAnimationEnd(Animator animation) {
                // Change the image resource
                imageView.setImageResource(newImageResource);

                // Start the second animation
                setIn.setTarget(imageView);
                setIn.start();
            }
        });

        // Start the first animation
        imageView.setImageResource(cardImageMap.get(backside));
        setOut.start();
    }

    private void animateCardPlay(int playerIndex, String playedCardName) {
        if(playerIndex != 0) {
            // Determine the hand and pile layouts and the tracking list for the player
            FrameLayout handLayout = playerIndex == 0 ? handLayout0 : handLayout1;
            LinearLayout pileLayout = playerIndex == 0 ? pileLayout0 : pileLayout1;
            List<ImageView> handCards = playerIndex == 0 ? player0Cards : player1Cards;

            // Find the card to play from the hand
            ImageView playedCard = null;
            for (ImageView card : cardsInTable) {
                String cardTag = (String) card.getTag();
                if (cardTag.equals(playedCardName)) {
                    playedCard = card;
                    break;
                }
            }

            if (playedCard != null) {
                playRandomCardSound();
                // Get the start position of the card in the hand layout
                int[] startPos = new int[2];
                playedCard.getLocationOnScreen(startPos);

                //    // Remove the card from the hand layout and tracking list
                handCards.remove(playedCard);
                relayHand(playerIndex);

                // Add the card to the pile layout
                handLayout.removeView(playedCard);
                playedCard.setElevation(pileLayout.getElevation());
                pileLayout.addView(playedCard);

                int parentWidth = pileLayout.getWidth();
                int parentHeight = pileLayout.getHeight();
                int pileCardHeight = parentHeight;
                int pileCardWidth = pileCardHeight * 2 / 3; // Example: Make card 1/5th of the layout width
                int pileOverlapMargin = (int) (pileCardWidth * (cardOffset));

                int totalWidth = pileCardWidth + (5 - 1) * (pileCardWidth - pileOverlapMargin);
                int padding = Math.max(0, (parentWidth - totalWidth) / 2);
                pileLayout.setPadding(padding, 0, 0, 0);

                LinearLayout.LayoutParams pileParams = new LinearLayout.LayoutParams(
                        pileCardWidth,
                        pileCardHeight
                );
                if (pileLayout.getChildCount() > 1) {
                    pileParams.setMargins(-pileOverlapMargin, 0, 0, 0); // Overlap in the pile
                }
                playedCard.setLayoutParams(pileParams);
                playedCard.setVisibility(View.INVISIBLE);

                // Declare a final reference for use in the lambda
                final ImageView finalPlayedCard = playedCard;

                flip_card_anim(playedCard, cardImageMap.get(playedCardName));

                // Allow layout updates to occur before calculating the new position
                pileLayout.post(() -> {
                    List<Integer> randOffsets = randomizePosition(finalPlayedCard);
                    // Get the new position of the card in the pile layout
                    int[] endPos = new int[2];
                    finalPlayedCard.getLocationOnScreen(endPos);

                    // Calculate translation distances
                    float translationX = startPos[0] - endPos[0];
                    float translationY = startPos[1] - endPos[1];

                    // Set initial position for the animation
                    finalPlayedCard.setTranslationX(translationX);
                    finalPlayedCard.setTranslationY(translationY);

                    // Animate the card to its final position with a rotation
                    finalPlayedCard.animate()
                            .translationX(randOffsets.get(1))
                            .translationY(randOffsets.get(2))
                            .rotation(randOffsets.get(0))
                            .setDuration(300)
                            .setInterpolator(new DecelerateInterpolator())
                            .withStartAction(new Runnable() {
                                @Override
                                public void run() {
                                    finalPlayedCard.setVisibility(View.VISIBLE);
                                }

                            })
                            .withEndAction(new Runnable() {
                                @Override
                                public void run() {
                                    finalPlayedCard.setVisibility(View.VISIBLE);
                                }
                            })
                            .start();
                });
            }
        }
    }

    private void relayHand(int playerIndex) {
        // Determine the hand and pile layouts and the tracking list for the player
        FrameLayout handLayout = playerIndex == 0 ? handLayout0 : handLayout1;
        List<ImageView> handCards = playerIndex == 0 ? player0Cards : player1Cards;

        // Angle for spreading the cards across the arc
        int numCards = 5;
        float startAngle = 100f;  // Starting angle of the arc (adjust as needed)
        float endAngle = 80f;     // Ending angle of the arc (adjust as needed)
        float angleStep = (endAngle - startAngle) / (numCards - 1);
        startAngle = (float) (startAngle + Math.floor((numCards - handCards.size()) / 2) * angleStep);
        float baseElevation = handLayout.getElevation();

        // Re-adjust the margins of the remaining cards in the hand layout with animation
        for (int i = 0; i < handCards.size(); i++) {
            final View card = handCards.get(i);
            card.setElevation(baseElevation + i * 1.0f);
            FrameLayout.LayoutParams params = (FrameLayout.LayoutParams) card.getLayoutParams();

            // Calculate new positions and rotation
            float angle = startAngle + i * angleStep;
            int[] arcPos = getArcPos_hand0(handLayout, angle);
            int newLeftMargin = arcPos[0] - params.width / 2;
            int newTopMargin;
            float newRotation;

            if (playerIndex == 0) {
                newTopMargin = arcPos[1];
                newRotation = -angle + 90;
            } else {
                newTopMargin = -arcPos[1];
                newRotation = angle - 90;
            }

            // Animate the card to the new position and rotation
            card.animate()
                    .translationX(newLeftMargin - params.leftMargin) // Difference in X position
                    .translationY(newTopMargin - params.topMargin) // Difference in Y position
                    .rotation(newRotation) // Animate to the new rotation
                    .setInterpolator(new AnticipateOvershootInterpolator()) // Smooth motion
                    .setDuration(300) // Animation duration
                    .withEndAction(() -> {
                        // Update the layout parameters at the end of the animation
                        params.leftMargin = newLeftMargin;
                        params.topMargin = newTopMargin;
                        card.setTranslationX(0); // Reset translation after applying margins
                        card.setTranslationY(0);
                        card.setLayoutParams(params);
                    })
                    .start();
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private void showHand(List<Player> gamePlayers, int playerIndex) {
        // Update the hand layout with the remaining cards
        List<Cards> currentHand = gamePlayers.get(playerIndex).getHand();
        List<ImageView> handCards = playerIndex == 0 ? player0Cards : player1Cards;
        FrameLayout handLayout = playerIndex == 0 ? handLayout0 : handLayout1;
        LinearLayout pileLayout = playerIndex == 0 ? pileLayout0 : pileLayout1;

        // Clear player's table
        pileLayout.removeAllViews();

        int parentWidth = handLayout.getWidth();
        int parentHeight = handLayout.getHeight();

        if (parentWidth > 0) {

            // Position cards to the handLayout
            int cardHeight = parentHeight;
            int cardWidth = parentHeight * 2 / 3;
            int numCards = 5;
            float startAngle = 100f;
            float endAngle = 80f;
            float angleStep = (endAngle - startAngle) / (numCards - 1);

            for (int i = handCards.size(); i < currentHand.size(); i++) {
                String cardName = currentHand.get(i).getSuit() + currentHand.get(i).getRank();
                if (cardImageMap.containsKey(cardName)) {
                    int cardDrawableRes = playerIndex == 0 ? cardImageMap.get(cardName) : cardImageMap.get(backside);

                    ImageView newCard = new ImageView(this);
                    newCard.setImageResource(cardDrawableRes);
                    newCard.setTag(cardName);
                    float angle = startAngle + i * angleStep;
                    int[] arcPos = getArcPos_hand0(handLayout, angle);

                    FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(cardWidth, cardHeight);
                    params.leftMargin = arcPos[0] - cardWidth / 2;
                    newCard.setPivotX(cardWidth / 2);
                    newCard.setPivotY(cardHeight / 2);
                    params.topMargin = (playerIndex == 0) ? arcPos[1] : -arcPos[1];
                    newCard.setRotation((playerIndex == 0) ? -angle + 90 : angle - 90);
                    newCard.setLayoutParams(params);

                    // Add card to hand layout and tracking list
                    handLayout.addView(newCard);
                    handCards.add(newCard);
                    cardsInTable.add(newCard);

                    // Enable Click and Drag & Drop for Player 0
                    if (playerIndex == 0) {
                        final Cards card = currentHand.get(i);

                        newCard.setOnTouchListener((v, event) -> {
                            if (event.getAction() == MotionEvent.ACTION_DOWN) {
                                if (!canUserInteract()) return false; // Early exit if bot is playing
                                // Check if newCard is in handCards and get its index
                                int cardIndex = handCards.indexOf(newCard);
                                if (cardIndex == -1) {
                                    return false;
                                }
                                if (!isPlayerTurn(playerIndex)) { // Check if it's the player's turn
                                    //Toast.makeText(this, "Wait for your turn!", Toast.LENGTH_SHORT).show();
                                } else if (game.roundNumber == 0) {
                                    //Toast.makeText(this, "Play bet first!", Toast.LENGTH_SHORT).show();
                                } else if (!(gamePlayers.get(playerIndex).validCards(game).contains(card))) {
                                    //Toast.makeText(this, "Follow round suit!", Toast.LENGTH_SHORT).show();
                                } else {

                                    playRandomCardSound();
                                    ClipData.Item item = new ClipData.Item((CharSequence) v.getTag());
                                    ClipData dragData = new ClipData(
                                            (CharSequence) v.getTag(),
                                            new String[]{ClipDescription.MIMETYPE_TEXT_PLAIN},
                                            item);
                                    View.DragShadowBuilder myShadow = new View.DragShadowBuilder(newCard);

                                    // Start the drag
                                    v.startDragAndDrop(dragData,  // The data to be dragged
                                            myShadow,  // The drag shadow builder
                                            v,      // No need to use local data
                                            0          // Flags. Not currently used, set to 0
                                    );
                                    v.setVisibility(View.VISIBLE); // Hide the card during drag
                                    draggedCardIndex = cardIndex;
                                    selectedCard = findCardByTag(newCard, currentHand);
                                    return true;
                                }
                            }
                            return false;
                        });
                    }
                }
            }

        } else {
            handLayout.post(() -> showHand(gamePlayers, playerIndex));
        }
    }

    private void playRandomCardSound() {
        Random rand = new Random();
        int randomSound = rand.nextInt(3);
        if (randomSound == 0) {
            soundPool.play(cardFlapSoundId1, 1.0f, 1.0f, 1, 0, 1.0f);
        }
        if (randomSound == 1) {
            soundPool.play(cardFlapSoundId2, 1.0f, 1.0f, 1, 0, 1.0f);
        }
        if (randomSound == 2) {
            soundPool.play(cardFlapSoundId3, 1.0f, 1.0f, 1, 0, 1.0f);
        }

    }
    private Cards findCardByTag(View draggedView, List<Cards> currentHand) {
        if (draggedView.getTag() == null) {
            return null; // No tag assigned to the view
        }

        String tag = draggedView.getTag().toString();

        for (Cards card : currentHand) {
            String cardString = (card.toString());
            if (tag.equals(cardString)) {
                return card;
            }
        }

        return null; // No matching card found
    }
    private void setCardDropCallbacks() {
        // Set the drag event listener for the handLayout0
        handLayout0.setOnDragListener((v, e) -> {
            switch (e.getAction()) {
                case DragEvent.ACTION_DRAG_STARTED:
                    return true;
                case DragEvent.ACTION_DROP:
                    Log.d("DragEvent", "Dropped home");
                    ImageView draggedView = (ImageView) e.getLocalState();
                    // Return card to hand
                    handLayout0.addView(draggedView);
                    player0Cards.add(draggedCardIndex,draggedView);
                    selectedCard = null; // Cancel play
                    relayHand(0);
                    raisePlayableCards();
                    return true;
                default:
                    return false;
            }
        });
        rootLayout.setOnDragListener((v, e) -> {
            switch (e.getAction()) {
                case DragEvent.ACTION_DRAG_STARTED:
                    Log.d("DragEvent", "STARTED");
                    lastTime = System.currentTimeMillis();;
                    lastX = e.getX();
                    lastY = e.getY();

                    ImageView draggedView2 = (ImageView) e.getLocalState();
                    // Remove from hand and update display
                    handLayout0.removeView(draggedView2);
                    player0Cards.remove(draggedView2);
                    relayHand(0); // Update hand display
                    lowerCards();

                    return true;
                case DragEvent.ACTION_DRAG_ENTERED:
                    Log.d("DragEvent", "ENTERED");
                    return true;
                case DragEvent.ACTION_DRAG_LOCATION:
                    long newTime = System.currentTimeMillis();
                    float newX = e.getX();
                    float newY = e.getY();

                    Log.d("DragDebug", "Location: x=" + newX + ", y=" + newY + ", time=" + newTime);

                    velocityX = (newX - lastX) / (newTime - lastTime);
                    velocityY = (newY - lastY) / (newTime - lastTime);
                    lastX = newX;
                    lastY = newY;
                    lastTime = newTime;
                    //Log.d("Fling debug","VelocityX: " + velocityX + "VelocityY: " + velocityY);

                    return true;
                case DragEvent.ACTION_DRAG_EXITED:
                    Log.d("DragEvent", "EXITED");
                    return true;
                case DragEvent.ACTION_DROP:
                    Log.d("DropDebug","Card dropped");
                    ImageView draggedView = (ImageView) e.getLocalState();
                    final float rootDropX = e.getX(); // Relative to rootLayout
                    final float rootDropY = e.getY();

                    draggedView.setElevation(pileLayout0.getElevation());
                    pileLayout0.addView(draggedView);
                    draggedView.setVisibility(View.INVISIBLE); // Start invisible to avoid flicker

                    // Convert root coordinates to pileLayout0 coordinates
                    int[] pileLocation = new int[2];
                    pileLayout0.getLocationOnScreen(pileLocation);
                    int[] rootLocation = new int[2];
                    rootLayout.getLocationOnScreen(rootLocation);
                    float pileDropX = rootDropX - (pileLocation[0] - rootLocation[0]);
                    float pileDropY = rootDropY - (pileLocation[1] - rootLocation[1]);

                    // Set initial position centered at drop point relative to pileLayout0
                    float initialX = pileDropX - draggedView.getWidth() / 2;
                    float initialY = pileDropY - draggedView.getHeight() / 2;
                    draggedView.setX(initialX);
                    draggedView.setY(initialY);
                    draggedView.setRotation(0);

                    // Calculate final position
                    int parentWidth = pileLayout0.getWidth();
                    int parentHeight = pileLayout0.getHeight();
                    int pileCardHeight = parentHeight;
                    int pileCardWidth = pileCardHeight * 2 / 3;
                    int pileOverlapMargin = (int) (pileCardWidth * (cardOffset));

                    int totalWidth = pileCardWidth + (5 - 1) * (pileCardWidth - pileOverlapMargin);
                    int padding = Math.max(0, (parentWidth - totalWidth) / 2);
                    pileLayout0.setPadding(padding, 0, 0, 0);

                    LinearLayout.LayoutParams pileParams = new LinearLayout.LayoutParams(
                            pileCardWidth,
                            pileCardHeight
                    );
                    if (pileLayout0.getChildCount() > 1) {
                        pileParams.setMargins(-pileOverlapMargin, 0, 0, 0); // Overlap with previous cards
                    }
                    draggedView.setLayoutParams(pileParams);

                    Log.d("Fling debug","VelocityX: " + velocityX + "VelocityY: " + velocityY);

                    float finalVelocityX = velocityX;
                    float finalVelocityY = velocityY;

                    pileLayout0.post(() -> {
                        List<Integer> randOffsets = randomizePosition(draggedView);
                        draggedView.setRotation(randOffsets.get(0));

                        // After layout, get the view’s assigned position
                        float left = draggedView.getLeft();  // Final X from LinearLayout
                        float top = draggedView.getTop();    // Final Y from LinearLayout

                        // Adjust translation to maintain initial drop position
                        draggedView.setTranslationX(initialX - left);
                        draggedView.setTranslationY(initialY - top);
                        draggedView.setVisibility(View.VISIBLE); // Show card immediately

                        // Calculate start velocities in pixels per second
                        float startVelocityX = finalVelocityX * 1000; // Convert from px/ms to px/s
                        float startVelocityY = finalVelocityY * 1000;

                        // Clip velocity to min and max
                        int min_velocity = 500;
                        int max_velocity = 5000;

                        if (startVelocityX > 0) {
                            startVelocityX = Math.max(min_velocity, Math.min(startVelocityX, max_velocity));
                        } else {
                            startVelocityX = Math.max(-max_velocity, Math.min(startVelocityX, -min_velocity));
                        }

                        if (startVelocityY > 0) {
                            startVelocityY = Math.max(min_velocity, Math.min(startVelocityY, max_velocity));
                        } else {
                            startVelocityY = Math.max(-max_velocity, Math.min(startVelocityY, -min_velocity));
                        }

                        // Create spring animation for X translation
                        SpringAnimation springX = new SpringAnimation(draggedView, DynamicAnimation.TRANSLATION_X, randOffsets.get(1));
                        springX.setStartVelocity(startVelocityX);
                        springX.getSpring().setStiffness(SpringForce.STIFFNESS_LOW); // 1500
                        springX.getSpring().setDampingRatio(SpringForce.DAMPING_RATIO_LOW_BOUNCY); // 0.5

                        // Create spring animation for Y translation
                        SpringAnimation springY = new SpringAnimation(draggedView, DynamicAnimation.TRANSLATION_Y, randOffsets.get(2));
                        springY.setStartVelocity(startVelocityY);
                        springY.getSpring().setStiffness(SpringForce.STIFFNESS_LOW);
                        springY.getSpring().setDampingRatio(SpringForce.DAMPING_RATIO_LOW_BOUNCY);

                        // Start both animations simultaneously
                        springX.start();
                        springY.start();

                    });
                    return true;
                case DragEvent.ACTION_DRAG_ENDED:
                    Log.d("DragEvent", "ENDED");
                    if (selectedCard != null) {
                        //isBotPlaying = true;
                        relayHand(0);
                        advanceGame();
                    }
                    return true;
                default:
                    return false;
            }
        });
    }

    private List<Integer> randomizePosition(ImageView v) {
        int viewWidth = v.getWidth();
        int offsetRange = (int) (viewWidth * 0.1);
        Random rand = new Random();
        int randomRotation = (int) ((rand.nextFloat() - 0.5f) * 7); // -3.5 to 3.5 degrees
        int randomXOffset = (int) ((rand.nextFloat() - 0.5f) * offsetRange); // -10 to 10 pixels
        int randomYOffset = (int) ((rand.nextFloat() - 0.5f) * offsetRange); // -5 to 5 pixels
        v.setTranslationX(randomXOffset);
        v.setTranslationY(randomYOffset);
        v.setRotation(randomRotation);

        List<Integer> list = new ArrayList<Integer>();
        list.add(randomRotation);
        list.add(randomXOffset);
        list.add(randomYOffset);
        return list;
    }

    private void raisePlayableCards() {
        List<ImageView> handCards = player0Cards;
        List<Cards> validCards = game.players.get(humanPlayerIndex).validCards(game);
        //if (validCards.size() < handCards.size()) {
        for (int i = 0; i < handCards.size(); i++) {
            ImageView cardView = handCards.get(i);
            Cards card = game.players.get(humanPlayerIndex).getHand().get(i);

            // Check if the card is playable
            if (validCards.contains(card)) {
                // Raise the card to indicate it's playable
                cardView.animate().translationY(-30).setDuration(400).setInterpolator(new OvershootInterpolator()); // Lift the card slightly
            } else {
                // Reset non-playable cards to their normal position
                cardView.animate().translationY(0).setDuration(400).setInterpolator(new DecelerateInterpolator());
            }
        }
        //}
    }

    private void lowerCards() {
        List<ImageView> handCards = player0Cards;
            for (int i = 0; i < handCards.size(); i++) {
                ImageView cardView = handCards.get(i);
                Cards card = game.players.get(humanPlayerIndex).getHand().get(i);
                    // Reset non-playable cards to their normal position
                cardView.animate().translationY(0).setDuration(400).setInterpolator(new DecelerateInterpolator());
            }
    }
}