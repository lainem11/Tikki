package com.example.tikki;

import androidx.annotation.DrawableRes;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.text.HtmlCompat;
import androidx.interpolator.view.animation.FastOutSlowInInterpolator;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.AnimatorSet;
import android.animation.ObjectAnimator;
import android.animation.PropertyValuesHolder;
import android.app.Dialog;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.drawable.AnimationDrawable;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.LayerDrawable;
import android.media.SoundPool;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.view.animation.DecelerateInterpolator;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    private ImageButton tikkiButton, kuolemaButton, chichiButton;
    private EditText playerName;
    private String enemyName = null;
    private SoundPool soundPool;
    private int cardSlideSoundId;
    private int playerCount = 2;
    private String difficulty;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tikkiButton = findViewById(R.id.tikki_button);
        kuolemaButton = findViewById(R.id.button_kuolema);
        chichiButton = findViewById(R.id.button_chichi);
        EditText textBar = findViewById(R.id.text_player_name);

        tikkiButton.setEnabled(false);
        fadeDrawable(tikkiButton, R.drawable.next_button_disabled,200);

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

        // Get parent container
        ViewGroup diffContainer = findViewById(R.id.diff_container);
        setExpandingCircleEffect(kuolemaButton,diffContainer);
        setExpandingCircleEffect(chichiButton,diffContainer);

        // Find views for each edge
        View topView = findViewById(R.id.pulseOverlayTop);
        View bottomView = findViewById(R.id.pulseOverlayBottom);
        View leftView = findViewById(R.id.pulseOverlayLeft);
        View rightView = findViewById(R.id.pulseOverlayRight);

        // Create pulse animations for each view
        ObjectAnimator topAnim = createPulseAnimator(topView);
        ObjectAnimator bottomAnim = createPulseAnimator(bottomView);
        ObjectAnimator leftAnim = createPulseAnimator(leftView);
        ObjectAnimator rightAnim = createPulseAnimator(rightView);

        // Add delays to create a sequential effect
        leftAnim.setStartDelay(2000);
        bottomAnim.setStartDelay(4000);
        rightAnim.setStartDelay(6000);

        // Combine all animations in a set
        AnimatorSet animatorSet = new AnimatorSet();
        animatorSet.playTogether(topAnim, leftAnim, bottomAnim, rightAnim);
        animatorSet.start();

        AnimationDrawable A1 = (AnimationDrawable) topView.getBackground();
        A1.setEnterFadeDuration(3000);
        A1.setExitFadeDuration(3000);
        A1.start();

        AnimationDrawable A2 = (AnimationDrawable) leftView.getBackground();
        A2.setEnterFadeDuration(3000);
        A2.setExitFadeDuration(3000);
        A2.start();

        AnimationDrawable A3 = (AnimationDrawable) bottomView.getBackground();
        A3.setEnterFadeDuration(3000);
        A3.setExitFadeDuration(3000);
        A3.start();

        AnimationDrawable A4 = (AnimationDrawable) rightView.getBackground();
        A4.setEnterFadeDuration(3000);
        A4.setExitFadeDuration(3000);
        A4.start();

        // Initialize SoundPool
        soundPool = new SoundPool.Builder()
                .setMaxStreams(5) // Allow up to 5 simultaneous sounds
                .build();
        cardSlideSoundId = soundPool.load(this, R.raw.card_slide, 1);

        tikkiButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                setButtonScaleAnimation(v);
                if (playerCount == 2) {
                    Intent intent = new Intent(MainActivity.this, GameActivity2p.class);
                    String enteredText = textBar.getText().toString();
                    if (enteredText.equals("")) {
                        enteredText = "P1";
                    }
                    intent.putExtra("playerName", enteredText);
                    intent.putExtra("enemyModel", enemyName);
                    startActivity(intent);
                } else if (playerCount == 4) {
                    Intent intent = new Intent(MainActivity.this, GameActivity4p.class);
                    String enteredText = textBar.getText().toString();
                    if (enteredText.equals("")) {
                        enteredText = "P1";
                    }
                    intent.putExtra("playerName", enteredText);
                    startActivity(intent);
                }
            }
        });

        kuolemaButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                soundPool.play(cardSlideSoundId, 1.0f, 1.0f, 1, 0, 1.0f);
                if (playerCount != 0) {
                    fadeDrawable(tikkiButton, R.drawable.next_button_enabled,200);
                    tikkiButton.setEnabled(true);
                }
                setButtonScaleAnimation(v);
                kuolemaButton.setImageResource(R.drawable.difficulty_button_hard_enabled);
                chichiButton.setImageResource(R.drawable.difficulty_button_easy_disabled);
                enemyName = "mobile_model_final_hard_test2.pt";
                difficulty = "hard";
            }
        });

        chichiButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                soundPool.play(cardSlideSoundId, 1.0f, 1.0f, 1, 0, 1.0f);
                if (playerCount != 0) {
                    fadeDrawable(tikkiButton, R.drawable.next_button_enabled,200);
                    tikkiButton.setEnabled(true);
                }
                setButtonScaleAnimation(v);
                kuolemaButton.setImageResource(R.drawable.difficulty_button_hard_disabled);
                chichiButton.setImageResource(R.drawable.difficulty_button_easy_enabled);
                enemyName = "mobile_model_easy.pt";
                difficulty = "easy";
            }
        });
    }

    private void setButtonScaleAnimation(View v) {
        ObjectAnimator scaleAnim = ObjectAnimator.ofPropertyValuesHolder(
                v,
                PropertyValuesHolder.ofFloat(View.SCALE_X, 1.0f, 0.9f, 1.0f),
                PropertyValuesHolder.ofFloat(View.SCALE_Y, 1.0f, 0.9f, 1.0f)
        );
        scaleAnim.setDuration(300); // Adjust as needed
        scaleAnim.setInterpolator(new DecelerateInterpolator());
        scaleAnim.start();
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
    private ObjectAnimator createPulseAnimator(View view) {
        ObjectAnimator animator = ObjectAnimator.ofFloat(view, "alpha", 0.1f, 0.5f, 0.1f);
        animator.setDuration(8000); // Animation duration in milliseconds
        animator.setRepeatCount(ObjectAnimator.INFINITE); // Loop the animation
        animator.setRepeatMode(ObjectAnimator.RESTART);
        return animator;
    }

    private void setExpandingCircleEffect(View button, ViewGroup parent) {
        button.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_DOWN) {
                createExpandingCircle(this, parent, event.getX(), event.getY(), v);
            }
            return false; // Ensures button click still works
        });
    }

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

    private GradientDrawable createCircleDrawable() {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setShape(GradientDrawable.OVAL);
        drawable.setColor(Color.parseColor("#66FFFFFF")); // Transparent white
        return drawable;
    }

    private void fadeDrawable(ImageButton button, @DrawableRes int newDrawableRes, int duration) {
        // Get the current drawable and the new drawable
        Drawable currentDrawable = button.getDrawable();
        Drawable newDrawable = getResources().getDrawable(newDrawableRes, null);

        // Create a LayerDrawable to overlay the current and new drawables
        LayerDrawable layerDrawable = new LayerDrawable(new Drawable[]{currentDrawable, newDrawable});
        button.setImageDrawable(layerDrawable);

        // Initially set the new drawable to be fully transparent
        newDrawable.setAlpha(0);

        // Animate the alpha transition
        ObjectAnimator fadeAnimator = ObjectAnimator.ofInt(newDrawable, "alpha", 0, 255);
        fadeAnimator.setDuration(duration); // Duration of the fade transition
        fadeAnimator.addListener(new AnimatorListenerAdapter() {
            @Override
            public void onAnimationEnd(Animator animation) {
                // Once the animation ends, remove the current drawable
                button.setImageDrawable(newDrawable);
            }
        });
        fadeAnimator.addUpdateListener(animation -> {
            // Keep the current drawable fully visible during the transition
            currentDrawable.setAlpha(255 - newDrawable.getAlpha());
        });

        fadeAnimator.start();
    }

}