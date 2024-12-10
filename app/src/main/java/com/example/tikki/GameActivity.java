package com.example.tikki;

import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.content.ContextCompat;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;

import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.w3c.dom.Text;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectStreamException;
import java.io.OutputStream;

import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.DecelerateInterpolator;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.util.Log;
import android.view.View;
import android.widget.Toast;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;

import java.util.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;


public class GameActivity extends AppCompatActivity {
    private Module module = null;
    private Tensor state = null;
    private float[] action;
    private int n_players = 2;
    private int round_n = 5;
    private int card_n = 5;
    private int n_bets = 3;
    private int n_inputs = card_n + n_players*(1+round_n) + n_players;
    private int n_outputs = 52+n_bets;
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
    private LinearLayout handLayout0,handLayout1,pileLayout0,pileLayout1;
    private View pulseOverlay0, pulseOverlay1;
    private Animation pulseAnimation;
    private HashMap<String, Integer> cardImageMap;
    private Button stateButton;
    private Button actionButton;
    private Button tikkiButton;
    private Button newGameButton,nextTurnButton,passButton,winButton,twoWinButton, quitButton;
    private ImageView selectedCardDrawable = null;
    private Cards selectedCard = null;
    private String selectedBet = null;
    private List<ImageView> player0Cards = new ArrayList<>();
    private List<ImageView> player1Cards = new ArrayList<>();
    private TextView score0,score1,text_player0Name,text_player1Name,player1_speech,text_game_end;
    private Object playedMove = null;
    private int humanPlayerIndex = 0;
    private int roundStarted = 0;
    private Handler handler;
    private String backside = "backside3";
    private boolean isBotPlaying = false; // Track if bot logic is active
    private final Random random = new Random(); // Shared random instance
    private int startingRound = 0;
    private int cardOffset = 25;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.tikki_main_screen);

        Intent intent = getIntent();
        String player0Name = intent.getStringExtra("playerName");
        String enemyModel = intent.getStringExtra("enemyModel");

        pulseAnimation = AnimationUtils.loadAnimation(this, R.anim.pulse);

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
        cardImageMap.put("♥11",R.drawable.jack_of_hearts2);
        cardImageMap.put("♥12",R.drawable.queen_of_hearts2);
        cardImageMap.put("♥13",R.drawable.king_of_hearts2);
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
        cardImageMap.put("♦11",R.drawable.jack_of_diamonds2);
        cardImageMap.put("♦12",R.drawable.queen_of_diamonds2);
        cardImageMap.put("♦13",R.drawable.king_of_diamonds2);
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
        cardImageMap.put("♣11",R.drawable.jack_of_clubs2);
        cardImageMap.put("♣12",R.drawable.queen_of_clubs2);
        cardImageMap.put("♣13",R.drawable.king_of_clubs2);
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
        cardImageMap.put("♠11",R.drawable.jack_of_spades2);
        cardImageMap.put("♠12",R.drawable.queen_of_spades2);
        cardImageMap.put("♠13",R.drawable.king_of_spades2);
        cardImageMap.put("♠14",R.drawable.ace_of_spades2);
        cardImageMap.put("backside3", R.drawable.backside3);

        try {
            module = LiteModuleLoader.load(assetFilePath(enemyModel));
            // Load the PyTorch model
            //module = Module.load(assetFilePath("model4.pt"));
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
        text_player0Name = findViewById(R.id.text_player0Name);
        text_player1Name = findViewById(R.id.text_player1Name);
        player1_speech = findViewById(R.id.player1_speech);

        nextTurnButton = findViewById(R.id.next_turn_button);
        newGameButton = findViewById(R.id.new_game_button);
        passButton = findViewById(R.id.pass_button);
        winButton = findViewById(R.id.win_button);
        twoWinButton = findViewById(R.id.two_win_button);
        quitButton = findViewById(R.id.buttonQuit);
        text_game_end = findViewById(R.id.text_game_end);

        pulseOverlay0.setVisibility(View.GONE);
        pulseOverlay1.setVisibility(View.GONE);
        passButton.setVisibility(View.GONE);
        winButton.setVisibility(View.GONE);
        twoWinButton.setVisibility(View.GONE);


        // Start tikki game
        String player1Name = "Bob";
        text_player0Name.setText(player0Name);
        text_player1Name.setText(player1Name);
        Player P1 = new Player(player0Name,null);
        Player P2 = new Player(player1Name,module);
        List<Player> players = new ArrayList<Player>();
        players.add(P1);
        players.add(P2);
        handler = new Handler(Looper.getMainLooper());
        game = startGame(players);


        passButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                selectedBet = "pass";
                passButton.setBackgroundColor(ContextCompat.getColor(GameActivity.this, R.color.button_pressed));
                winButton.setBackgroundColor(ContextCompat.getColor(GameActivity.this, R.color.button_default));
                twoWinButton.setBackgroundColor(ContextCompat.getColor(GameActivity.this, R.color.button_default));
                nextTurnButton.setEnabled(true);
            }
        });

        winButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                selectedBet = "win";
                passButton.setBackgroundColor(ContextCompat.getColor(GameActivity.this, R.color.button_default));
                winButton.setBackgroundColor(ContextCompat.getColor(GameActivity.this, R.color.button_pressed));
                twoWinButton.setBackgroundColor(ContextCompat.getColor(GameActivity.this, R.color.button_default));
                nextTurnButton.setEnabled(true);
            }
        });

        twoWinButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                selectedBet = "2-win";
                passButton.setBackgroundColor(ContextCompat.getColor(GameActivity.this, R.color.button_default));
                winButton.setBackgroundColor(ContextCompat.getColor(GameActivity.this, R.color.button_default));
                twoWinButton.setBackgroundColor(ContextCompat.getColor(GameActivity.this, R.color.button_pressed));
                nextTurnButton.setEnabled(true);
            }
        });

        nextTurnButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                lowerCards();
                advanceGame();
                if (game.gameOver) {
                    endGame();
                }
            }
        });

        newGameButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                game = startGame(players);
            }
        });

        quitButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Intent intent = new Intent(GameActivity.this, MainActivity.class);
                startActivity(intent);
            }
        });
    }

    public void start_round() {
        // Reset the round state and show all hands
        for (int i = 0; i < game.players.size(); i++) {
            showHand(game.players, i);
        }
        roundStarted = 1;
        updateTurnIndicator(game.currentPlayerIndex);
        resetSelection();

        // Bot moves should only occur after human input during betting or card play
        //if (game.currentPlayerIndex != humanPlayerIndex) {
        //    play_bot_moves();
        //}
        showBetButtons();
        //handler.removeCallbacksAndMessages(null);
        isBotPlaying = false;
    }

    private void play_bot_moves() {
        if (game != null && game.currentPlayerIndex != humanPlayerIndex && !isBotPlaying) {
            isBotPlaying = true;
            // Generate a random delay using normal distribution
            int randomDelay = getNormalDelay(500, 300); // Mean = 300ms, Standard deviation = 100ms

            // Delay the bot's move to simulate thinking
            handler.postDelayed(() -> {
                if (game != null && game.currentPlayerIndex != humanPlayerIndex) {
                    advanceGame(); // Perform the bot's move
                }
                isBotPlaying = false; // Reset flag after bot move
                play_bot_moves(); // Check for the next move
            }, randomDelay); // Delay in milliseconds
        }
        if (game.currentPlayerIndex == humanPlayerIndex) {
            raisePlayableCards();
        }

    }

    private int getNormalDelay(int mean, int stdDev) {
        double gaussian = random.nextGaussian(); // Generate a random value from a standard normal distribution
        int delay = (int) (mean + gaussian * stdDev); // Scale and shift

        // Clamp delay to a valid range (e.g., 100ms to 1000ms)
        return Math.max(300, Math.min(1000, delay));
    }
    private void advanceGame() {
        Log.e("Advancing game","advancing game");
        if (game != null) {
            if (!game.gameOver) {
                // Proceed with the next turn if game is not over
                int cardPlayer = game.currentPlayerIndex;
                int oldRoundNumber = game.roundNumber;

                // Check if move is based on selection
                if (selectedBet != null) {
                    playedMove = selectedBet;
                    hideBetButtons();
                } else if (selectedCard != null) {
                    playedMove = selectedCard;
                } else {
                    playedMove = null;
                    Log.e("Non-player turn","No bet or card selected");
                }
                resetSelection();

                // Play turn and update visual elements
                Object[] result = game.nextTurn(playedMove);
                String moveString = (String) result[0];
                setScores();
                updateTurnIndicator(game.currentPlayerIndex);

                // If card was played (not round 0), show card play animation
                if (oldRoundNumber != 0) {
                    updateHand(game.players,cardPlayer,moveString);
                    roundStarted = 0;
                }

                // Start next round if round ended on this turn
                if (game.roundNumber == 0 && roundStarted == 0) {
                    startingRound = 1;
                    hideTurnIndicator();
                    handler.postDelayed(() -> {
                        start_round();
                        startingRound = 0;
                    }, 2000); // 2000 ms = 2 seconds delay
                }

                // Update bet speech
                if (oldRoundNumber == 0 && cardPlayer != humanPlayerIndex) {
                    if (startingRound == 1) {
                        handler.postDelayed(() -> {
                        player1_speech.setText(getString(R.string.player1_speech, moveString));
                        }, 2000); // 2000 ms = 2 seconds delay
                    } else {
                        player1_speech.setText(getString(R.string.player1_speech, moveString));
                    }
                } else {
                    player1_speech.setText("");
                }
                play_bot_moves();
            } else {
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
        newGameButton.setVisibility(View.VISIBLE);
        hideTurnIndicator();
        text_game_end.setText(getString(R.string.winner_name, game.gameWinner.name));
        text_game_end.setVisibility(View.VISIBLE);
    }

    private void setScores() {
        Map<Player, Integer> playerScores = game.playerScores;
        List<Player> players = game.players;
        score0.setText(playerScores.get(players.get(0)).toString());
        score1.setText(playerScores.get(players.get(1)).toString());
    }

    private Tikki startGame(List<Player> players) {
        game = new Tikki(players);
        game.newGame();
        newGameButton.setVisibility(View.GONE);
        text_game_end.setVisibility(View.GONE);
        start_round();
        setScores();
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
    }

    // Helper method to check if it's the player's turn
    private boolean isPlayerTurn(int playerIndex) {
        return game.currentPlayerIndex == playerIndex;
    }

    // Reset selection after playing the card
    private void resetSelection() {
        selectedCardDrawable = null;
        selectedCard = null;
        selectedBet = null;
        passButton.setBackgroundColor(ContextCompat.getColor(GameActivity.this, R.color.button_default));
        winButton.setBackgroundColor(ContextCompat.getColor(GameActivity.this, R.color.button_default));
        twoWinButton.setBackgroundColor(ContextCompat.getColor(GameActivity.this, R.color.button_default));
        nextTurnButton.setEnabled(false);
    }

    private void updateTurnIndicator(int currentPlayerIndex) {
        // Stop animations on both layouts
        pulseOverlay0.clearAnimation();
        pulseOverlay1.clearAnimation();

        // Reset background for both hand layouts
        //handLayout0.setBackgroundResource(0);
        //handLayout1.setBackgroundResource(0);

        // Start the animation on the current player's hand layout
        if (currentPlayerIndex == 0) {
            pulseOverlay0.setVisibility(View.VISIBLE);
            pulseOverlay0.startAnimation(pulseAnimation); // Start pulse animation on player 0 overlay
            pulseOverlay1.setVisibility(View.GONE);
        } else if (currentPlayerIndex == 1) {
            pulseOverlay1.setVisibility(View.VISIBLE);
            pulseOverlay1.startAnimation(pulseAnimation); // Start pulse animation on player 1 overlay
            pulseOverlay0.setVisibility(View.GONE);
        }
    }

    private void updateHand(List<Player> gamePlayers, int playerIndex, String playedCardName) {
        // Determine the hand and pile layouts and the tracking list for the player
        LinearLayout handLayout = playerIndex == 0 ? handLayout0 : handLayout1;
        LinearLayout pileLayout = playerIndex == 0 ? pileLayout0 : pileLayout1;
        List<ImageView> handCards = playerIndex == 0 ? player0Cards : player1Cards;

        // Find the card to play from the hand
        ImageView playedCard = null;
        for (ImageView card : handCards) {
            String cardTag = (String) card.getTag();
            if (cardTag.equals(playedCardName)) {
                playedCard = card;
                break;
            }
        }

        if (playedCard != null) {
            // Get the start position of the card in the hand layout
            int[] startPos = new int[2];
            playedCard.getLocationOnScreen(startPos);

            // Remove the card from the hand layout and tracking list
            handLayout.removeView(playedCard);
            handCards.remove(playedCard);

            // Adjust layout for centering remaining cards
            int overlapMargin = (int) (cardOffset * getResources().getDisplayMetrics().density);

            // Re-center handpile
            //int parentWidth = handLayout.getWidth();
            //int cardWidth = parentWidth / 5;
            //int totalWidth = cardWidth + (handCards.size() - 1) * (cardWidth - overlapMargin);
            //int padding = Math.max(0, (parentWidth - totalWidth) / 2);
            //handLayout.setPadding(padding, 0, padding, 0);

            // Re-adjust the margins of the remaining cards in the hand layout
            for (int i = 0; i < handCards.size(); i++) {
                LinearLayout.LayoutParams layoutParams = (LinearLayout.LayoutParams) handCards.get(i).getLayoutParams();
                if (i > 0) {
                    layoutParams.setMargins(-overlapMargin, 0, 0, 0); // Maintain overlap for subsequent cards
                } else {
                    layoutParams.setMargins(0, 0, 0, 0); // No overlap for the first card
                }
                handCards.get(i).setLayoutParams(layoutParams);
            }

            // Add the card to the pile layout with overlap
            pileLayout.addView(playedCard);
            int pileOverlapMargin = (int) (cardOffset * getResources().getDisplayMetrics().density); // Adjust overlap for the pile
            LinearLayout.LayoutParams pileParams = new LinearLayout.LayoutParams(
                    playedCard.getLayoutParams().width,
                    playedCard.getLayoutParams().height
            );
            if (pileLayout.getChildCount() > 1) {
                pileParams.setMargins(-pileOverlapMargin, 0, 0, 0); // Overlap in the pile
            }
            playedCard.setLayoutParams(pileParams);
            int parentWidth = pileLayout.getWidth();
            // Calculate the width for each card (one fifth of the parent layout width)
            int cardWidth = parentWidth / 5;
            int totalWidth = cardWidth + (5-1) * (cardWidth - pileOverlapMargin);
            int padding = Math.max(0, (parentWidth - totalWidth) / 2);
            pileLayout.setPadding(padding, 0, padding, 0);

            playedCard.setVisibility(View.INVISIBLE);

            // Declare a final reference for use in the lambda
            final ImageView finalPlayedCard = playedCard;

            // Allow layout updates to occur before calculating the new position
            pileLayout.post(() -> {
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
                        .translationX(0)
                        .translationY(0)
                        .rotationY(180)  // This rotates the card 180 degrees along the Y-axis
                        .setDuration(300)
                        .withStartAction(new Runnable() {
                            @Override
                            public void run() {
                                // Set the visibility to VISIBLE after the animation ends
                                finalPlayedCard.setVisibility(View.VISIBLE);
                            }
                        })
                        .withEndAction(new Runnable() {
                            @Override
                            public void run() {
                                // Ensure the card is flipped back to 0 degrees after the animation finishes
                                finalPlayedCard.setRotationY(0);
                                // Show card face up
                                finalPlayedCard.setImageResource(cardImageMap.get(playedCardName));
                            }
                        })
                        .start();
            });
        }
    }

    private void showHand2(List<Player> gamePlayers, int playerIndex) {
        List<Cards> currentHand = gamePlayers.get(playerIndex).getHand();
        List<ImageView> handCards = playerIndex == 0 ? player0Cards : player1Cards;
        LinearLayout handLayout = playerIndex == 0 ? handLayout0 : handLayout1;
        LinearLayout pileLayout = playerIndex == 0 ? pileLayout0 : pileLayout1;

        // Clear player's table
        pileLayout.removeAllViews();

        int parentWidth = handLayout.getWidth();

        if (parentWidth > 0) {
            int cardWidth = parentWidth / 5;
            int overlapMargin = (int) (cardOffset * getResources().getDisplayMetrics().density);
            int fanSpread = Math.min(15, currentHand.size() * 2); // Total angle spread for the fan (in degrees)
            int angleStep = currentHand.size() > 1 ? fanSpread / (currentHand.size() - 1) : 0;
            int centerOffset = (parentWidth - (currentHand.size() * (cardWidth - overlapMargin))) / 2;

            handLayout.setPadding(centerOffset, 0, centerOffset, 0);

            for (int i = handCards.size(); i < currentHand.size(); i++) {
                String cardName = currentHand.get(i).getSuit() + currentHand.get(i).getRank();
                if (cardImageMap.containsKey(cardName)) {
                    int cardDrawableRes = (playerIndex == 0) ?
                            cardImageMap.get(cardName) :
                            cardImageMap.get(backside);

                    ImageView newCard = new ImageView(this);
                    newCard.setImageResource(cardDrawableRes);
                    newCard.setTag(cardName);

                    // Calculate rotation and translation
                    int rotationAngle = -fanSpread / 2 + i * angleStep; // Spread from -fanSpread/2 to +fanSpread/2
                    float translationY = (float) Math.abs(rotationAngle) * -3; // Slightly raise the cards based on the angle

                    // Set layout parameters
                    LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(
                            cardWidth,
                            LinearLayout.LayoutParams.WRAP_CONTENT
                    );
                    if (i > 0) {
                        layoutParams.setMargins(-overlapMargin, 0, 0, 0); // Overlap by setting a negative left margin
                    }
                    newCard.setLayoutParams(layoutParams);

                    // Apply rotation and translation
                    newCard.setPivotX(cardWidth / 2f);
                    newCard.setPivotY(newCard.getHeight());
                    newCard.setRotation(rotationAngle);
                    newCard.setTranslationY(translationY);

                    // Add click listener for player 0
                    if (playerIndex == 0) {
                        final int finalIndex = i;
                        final Cards card = currentHand.get(finalIndex);
                        newCard.setOnClickListener(view -> handleCardClick(newCard, card, playerIndex));
                    }

                    // Add the card to the hand layout and tracking list
                    handLayout.addView(newCard);
                    handCards.add(newCard);
                }
            }
        } else {
            // Handle the case where the layout width is not yet determined
            handLayout.post(() -> showHand(gamePlayers, playerIndex));
        }
    }

    // Card click handler
    private void handleCardClick(ImageView cardView, Cards card, int playerIndex) {
        if (!isPlayerTurn(playerIndex)) {
            Toast.makeText(this, "Not your turn!", Toast.LENGTH_SHORT).show();
            return;
        }
        if (selectedCardDrawable != null) {
            selectedCardDrawable.animate().translationY(0).setDuration(200);
        }
        if (selectedCardDrawable != cardView) {
            selectedCardDrawable = cardView;
            selectedCard = card;
            nextTurnButton.setEnabled(true);
            cardView.animate().translationY(-20).setDuration(200);
        } else {
            resetSelection();
        }
    }

    private void showHand(List<Player> gamePlayers, int playerIndex) {
        // Update the hand layout with the remaining cards
        List<Cards> currentHand = gamePlayers.get(playerIndex).getHand();
        List<ImageView> handCards = playerIndex == 0 ? player0Cards : player1Cards;
        LinearLayout handLayout = playerIndex == 0 ? handLayout0 : handLayout1;
        LinearLayout pileLayout = playerIndex == 0 ? pileLayout0 : pileLayout1;

        // Clear player's table
        pileLayout.removeAllViews();

        int parentWidth = handLayout.getWidth();

        if (parentWidth > 0) {
            // Calculate the width for each card (one fifth of the parent layout width)
            int cardWidth = parentWidth / 5;
            int overlapMargin = (int) (cardOffset * getResources().getDisplayMetrics().density);
            int totalWidth = cardWidth + (currentHand.size() - 1) * (cardWidth - overlapMargin);
            int padding = Math.max(0, (parentWidth - totalWidth) / 2);
            handLayout.setPadding(padding, 0, padding, 0);

            for (int i = handCards.size(); i < currentHand.size(); i++) {
                // Add only the missing cards
                String cardName = currentHand.get(i).getSuit() + currentHand.get(i).getRank();
                if (cardImageMap.containsKey(cardName)) {
                    int cardDrawableRes;
                    if (playerIndex == 0) {
                        cardDrawableRes = cardImageMap.get(cardName);
                    } else {
                        // plot backside
                        cardDrawableRes = cardImageMap.get(backside);
                    }

                    ImageView newCard = new ImageView(this);
                    newCard.setImageResource(cardDrawableRes);
                    newCard.setTag(cardName); // Tag the card for identification

                    // Set layout parameters
                    LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(
                            cardWidth,
                            LinearLayout.LayoutParams.WRAP_CONTENT
                    );
                    if (i > 0) {
                        layoutParams.setMargins(-overlapMargin, 0, 0, 0); // Overlap by setting a negative left margin
                    }
                    newCard.setLayoutParams(layoutParams);

                    // Set up the card click listener for human player
                    if (playerIndex == 0) {
                        final int finalIndex = i;
                        final Cards card = currentHand.get(finalIndex);
                        newCard.setOnClickListener(view -> {
                            if (!isPlayerTurn(playerIndex)) { // Check if it's the player's turn
                                Toast.makeText(this, "Not your turn!", Toast.LENGTH_SHORT).show();
                            } else if (game.roundNumber == 0) {
                                Toast.makeText(this, "Play bet first!", Toast.LENGTH_SHORT).show();
                            } else if (!(gamePlayers.get(playerIndex).validCards(game).contains(card))) {
                                Toast.makeText(this, "Illegal move!", Toast.LENGTH_SHORT).show();
                            } else {
                                // Deselect the previously selected card, if any
                                if (selectedCardDrawable != null) {
                                    selectedCardDrawable.animate().translationY(-20).setDuration(200); // Lower the previous card
                                }
                                if (selectedCardDrawable != newCard) {
                                    // Select the current card
                                    selectedCardDrawable = newCard;
                                    selectedCard = card;
                                    nextTurnButton.setEnabled(true);
                                    //selectedCardIndex = finalIndex;//.indexOf(currentHand.get(finalIndex));
                                    newCard.animate().translationY(-40).setDuration(200); // Lift the card slightly
                                } else {
                                    resetSelection();
                                }
                            }
                        });
                    }
                    // Add the card to the hand layout and tracking list
                    handLayout.addView(newCard);
                    handCards.add(newCard);
                }
            }
        } else {
            // Handle the case where the layout width is not yet determined
            // For example, you might need to post this update to a handler or wait for layout completion
            handLayout.post(() -> showHand(gamePlayers, playerIndex));
        }
    }

    /**
     * Raises cards that can be played by the player, visually indicating their availability.
     */
    private void raisePlayableCards() {
        List<ImageView> handCards = player0Cards;
        List<Cards> validCards = game.players.get(humanPlayerIndex).validCards(game);
        if (validCards.size() < handCards.size()) {
            for (int i = 0; i < handCards.size(); i++) {
                ImageView cardView = handCards.get(i);
                Cards card = game.players.get(humanPlayerIndex).getHand().get(i);

                // Check if the card is playable
                if (game.players.get(humanPlayerIndex).validCards(game).contains(card)) {
                    // Raise the card to indicate it's playable
                    cardView.animate().translationY(-20).setDuration(200); // Lift the card slightly
                } else {
                    // Reset non-playable cards to their normal position
                    cardView.animate().translationY(0).setDuration(200);
                }
            }
        }
    }
    private void lowerCards() {
        List<ImageView> handCards = player0Cards;
            for (int i = 0; i < handCards.size(); i++) {
                ImageView cardView = handCards.get(i);
                Cards card = game.players.get(humanPlayerIndex).getHand().get(i);
                    // Reset non-playable cards to their normal position
                cardView.animate().translationY(0).setDuration(200);
            }
    }
}