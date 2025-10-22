:- discontiguous ability_of/2.
:- discontiguous in_hand/2.
:- discontiguous in_deck/2.
:- discontiguous in_board/2.

% -----------------------
% ИНИЦИАЛЬНОЕ СОСТОЯНИЕ
% Ход: CurrentPlayer = 1, Turn = 1
% Оба игрока: Mana = 1, MaxMana = 1
% Доски пусты
% -----------------------
initial_state(game_state(
    player(30, 1, 1,                   % Player1: Health=30, Mana=1, MaxMana=1
           [ card(stonetusk_boar,1,1,1), card(bloodfen_raptor,2,3,2),
             card(chillwind_yeti,4,4,5), card(loot_hoarder,2,2,1),
             card(argent_squire,1,1,1), card(war_golem,7,7,7),
             card(silverhand_recruit,1,1,1), card(acolyte_of_pain,3,1,3),
             card(fire_elemental,6,6,5), card(leeroy_jenkins,5,6,2)
           ], % Deck (Player1)
           [ card(wisp,0,1,1), card(murloc_raider,1,2,1) ], % Hand (Player1)
           []),   % Board пустой (Player1)
    player(30, 1, 1,                   % Player2: Health=30, Mana=1, MaxMana=1
           [ card(scarlet_crusader,3,3,1), card(river_crocolisk,2,2,3),
             card(boulderfist_ogre,6,6,7), card(chillwind_yeti,4,4,5),
             card(loot_hoarder,2,2,1), card(acolyte_of_pain,3,1,3),
             card(war_golem,7,7,7)
           ], % Deck (Player2)
           [ card(river_crocolisk,2,2,3), card(boulderfist_ogre,6,6,7) ], % Hand (Player2)
           []),  % Board пустой (Player2)
    1, 1  % CurrentPlayer = 1, Turn = 1
)).

% -----------------------
% КАРТЫ (card/4: Name, Cost, Attack, Health)
% Все характеристики хранятся в card/4; дополнительные предикаты выведены из него.
% -----------------------
card(wisp, 0, 1, 1).
card(stonetusk_boar, 1, 1, 1).
card(murloc_raider, 1, 2, 1).
card(bloodfen_raptor, 2, 3, 2).
card(river_crocolisk, 2, 2, 3).
card(chillwind_yeti, 4, 4, 5).
card(boulderfist_ogre, 6, 6, 7).
card(argent_squire, 1, 1, 1).
card(scarlet_crusader, 3, 3, 1).
card(loot_hoarder, 2, 2, 1).
card(war_golem, 7, 7, 7).
card(acolyte_of_pain, 3, 1, 3).
card(leeroy_jenkins, 5, 6, 2).
card(silverhand_recruit, 1, 1, 1).
card(fire_elemental, 6, 6, 5).
card(novice_engineer, 2, 1, 1).
card(murloc_scout, 1, 1, 1).
card(blood_mage_thalnos, 2, 1, 1).
card(senjin_shieldmasta, 4, 3, 5).

% -----------------------
% ОДНОАРГУМЕНТНЫЕ ФАКТЫ (>=20)
% Описывают имена, роли, способности, трибы, и т.д.
% -----------------------
minion_name(wisp).
minion_name(stonetusk_boar).
minion_name(murloc_raider).
minion_name(bloodfen_raptor).
minion_name(river_crocolisk).
minion_name(chillwind_yeti).
minion_name(boulderfist_ogre).
minion_name(argent_squire).
minion_name(scarlet_crusader).
minion_name(loot_hoarder).
minion_name(war_golem).
minion_name(acolyte_of_pain).
minion_name(leeroy_jenkins).
minion_name(silverhand_recruit).
minion_name(fire_elemental).
minion_name(novice_engineer).
minion_name(murloc_scout).
minion_name(blood_mage_thalnos).
minion_name(senjin_shieldmasta).

player_name(player1).
player_name(player2).

ability(charge).
ability(taunt).
ability(rush).
ability(divine_shield).
ability(battlecry).
ability(deathrattle).

tribe(beast).
tribe(murloc).
tribe(mech).
tribe(elemental).

rarity(common).
rarity(rare).

zone(hand).
zone(deck).
zone(board).

role(hero).

expansion(core).
expansion(classic).

% -----------------------
% ДВУХАРГУМЕНТНЫЕ ФАКТЫ (10-15+)
% способности, трибы, положение карт в зонах (синхронизировано с initial_state)
% -----------------------

% способности у карт
ability_of(stonetusk_boar, charge).
ability_of(argent_squire, divine_shield).
ability_of(murloc_raider, rush).
ability_of(scarlet_crusader, taunt).
ability_of(leeroy_jenkins, charge).
ability_of(loot_hoarder, deathrattle).

% трибы
tribe_of(murloc_raider, murloc).
tribe_of(river_crocolisk, beast).
tribe_of(fire_elemental, elemental).

% положение карт (дополнительные факты, полезны для запросов)
% Синхронизированы с initial_state: in_hand соответствует Hand, in_deck — Deck, in_board — пусто
in_hand(player1, wisp).
in_hand(player1, murloc_raider).
in_hand(player2, river_crocolisk).
in_hand(player2, boulderfist_ogre).

in_deck(player1, stonetusk_boar).
in_deck(player1, bloodfen_raptor).
in_deck(player1, chillwind_yeti).
in_deck(player1, loot_hoarder).
in_deck(player1, argent_squire).
in_deck(player1, war_golem).
in_deck(player1, silverhand_recruit).
in_deck(player1, acolyte_of_pain).
in_deck(player1, fire_elemental).
in_deck(player1, leeroy_jenkins).

in_deck(player2, scarlet_crusader).
in_deck(player2, river_crocolisk).
in_deck(player2, boulderfist_ogre).
in_deck(player2, chillwind_yeti).
in_deck(player2, loot_hoarder).
in_deck(player2, acolyte_of_pain).
in_deck(player2, war_golem).

% доски пусты (нет in_board фактов)
% in_board(player1, ...).  % нет
% in_board(player2, ...).  % нет

% -----------------------
% ВСПОМОГАТЕЛЬНЫЕ ПРЕДИКАТЫ (утилиты)
% -----------------------
% remove_from_hand(HandList, CardTerm, NewHandList)
remove_from_hand(Hand, Card, NewHand) :-
    select(Card, Hand, NewHand).

% replace(Index, List, NewItem, NewList)
replace(Index, List, NewItem, NewList) :-
    length(Prefix, Index1),
    Index is Index1 + 1,
    append(Prefix, [_|Suffix], List),
    append(Prefix, [NewItem|Suffix], NewList).

% Новая форма minion: minion(Name,Attack,Health,CanAttack,RushOnly)
% is_dead для minion(Name,Attack,Health,CanAttack,RushOnly)
is_dead(minion(_, _, Health, _, _)) :- Health =< 0.
remove_dead_minions(Board, FinalBoard) :-
    exclude(is_dead, Board, FinalBoard).

% draw_card(Deck, Hand, NewDeck, NewHand)
draw_card([Card|Deck], Hand, Deck, [Card|Hand]).
draw_card([], Hand, [], Hand).

% update_minion_health(Board, Index, NewHealth, NewBoard)
update_minion_health(Board, Index, NewHealth, NewBoard) :-
    nth1(Index, Board, minion(Name, Attack, _OldH, CanAttack, RushOnly)),
    replace(Index, Board, minion(Name, Attack, NewHealth, CanAttack, RushOnly), NewBoard).

% set_all_minions_ready (CanAttack = true, RushOnly -> false (снимаем ограничение рывка на новый ход))
set_minions_ready([], []).
set_minions_ready([minion(Name, Attack, Health, _OldCan, _OldRush)|T],
                  [minion(Name, Attack, Health, true, false)|T2]) :-
    set_minions_ready(T, T2).

% -----------------------
% ВЫВОД ХАРАКТЕРИСТИК ИЗ card/4
% вместо дублирования facts используем правила, это убирает предупреждения о discontiguous:
% -----------------------
card_cost(Name, Cost) :- card(Name, Cost, _, _).
card_attack(Name, Attack) :- card(Name, _, Attack, _).
card_health(Name, Health) :- card(Name, _, _, Health).

% -----------------------
% ПРЕДИКАТЫ ИГРОВОГО ПРОЦЕССА (play/attack/end_turn)
% Примечание: упрощённая симуляция для демонстрации логики.
% -----------------------

% --- play_minion: для CurrentPlayer = 1
play_minion(GameState, MinionName, NewGameState) :-
    GameState = game_state(P1, P2, 1, Turn),
    P1 = player(H1, M1, MaxM1, Deck1, Hand1, Board1),
    member(card(MinionName, Cost, Attack, Health), Hand1),
    Cost =< M1,
    NewMana is M1 - Cost,
    remove_from_hand(Hand1, card(MinionName, Cost, Attack, Health), NewHand1),
    ( ability_of(MinionName, charge) -> (CanAttack = true, RushOnly = false)
    ; ability_of(MinionName, rush)   -> (CanAttack = true, RushOnly = true)
    ; (CanAttack = false, RushOnly = false)
    ),
    NewBoard1 = [minion(MinionName, Attack, Health, CanAttack, RushOnly)|Board1],
    NewP1 = player(H1, NewMana, MaxM1, Deck1, NewHand1, NewBoard1),
    NewGameState = game_state(NewP1, P2, 1, Turn).

% --- play_minion: для CurrentPlayer = 2
play_minion(GameState, MinionName, NewGameState) :-
    GameState = game_state(P1, P2, 2, Turn),
    P2 = player(H2, M2, MaxM2, Deck2, Hand2, Board2),
    member(card(MinionName, Cost, Attack, Health), Hand2),
    Cost =< M2,
    NewMana is M2 - Cost,
    remove_from_hand(Hand2, card(MinionName, Cost, Attack, Health), NewHand2),
    ( ability_of(MinionName, charge) -> (CanAttack = true, RushOnly = false)
    ; ability_of(MinionName, rush)   -> (CanAttack = true, RushOnly = true)
    ; (CanAttack = false, RushOnly = false)
    ),
    NewBoard2 = [minion(MinionName, Attack, Health, CanAttack, RushOnly)|Board2],
    NewP2 = player(H2, NewMana, MaxM2, Deck2, NewHand2, NewBoard2),
    NewGameState = game_state(P1, NewP2, 2, Turn).

% --- attack: для CurrentPlayer = 1
% TargetIndex = 0 => атака в героя противника
attack(GameState, AttackerIndex, TargetIndex, NewGameState) :-
    GameState = game_state(P1, P2, 1, Turn),
    P1 = player(H1, M1, MaxM1, Deck1, Hand1, Board1),
    P2 = player(H2, M2, MaxM2, Deck2, Hand2, Board2),
    nth1(AttackerIndex, Board1, minion(AttackerName, AttackerAttack, AttackerHealth, AttackerCanAttack, AttackerRushOnly)),
    ( AttackerCanAttack = true -> true ; fail ),
    ( AttackerRushOnly = true, TargetIndex = 0 -> fail ; true ),
    ( TargetIndex = 0 ->
        NewH2 is H2 - AttackerAttack,
        replace(AttackerIndex, Board1, minion(AttackerName, AttackerAttack, AttackerHealth, false, false), TempBoard1),
        NewP2 = player(NewH2, M2, MaxM2, Deck2, Hand2, Board2),
        NewP1 = player(H1, M1, MaxM1, Deck1, Hand1, TempBoard1),
        NewGameState = game_state(NewP1, NewP2, 1, Turn)
    ;
        nth1(TargetIndex, Board2, minion(TargetName, TargetAttack, TargetHealth, TargetCanAttack, TargetRushOnly)),
        NewAttackerHealth is AttackerHealth - TargetAttack,
        NewTargetHealth is TargetHealth - AttackerAttack,
        replace(AttackerIndex, Board1, minion(AttackerName, AttackerAttack, NewAttackerHealth, false, false), TempBoard1),
        replace(TargetIndex, Board2, minion(TargetName, TargetAttack, NewTargetHealth, TargetCanAttack, TargetRushOnly), TempBoard2),
        remove_dead_minions(TempBoard1, FinalBoard1),
        remove_dead_minions(TempBoard2, FinalBoard2),
        NewP1 = player(H1, M1, MaxM1, Deck1, Hand1, FinalBoard1),
        NewP2 = player(H2, M2, MaxM2, Deck2, Hand2, FinalBoard2),
        NewGameState = game_state(NewP1, NewP2, 1, Turn)
    ).

% --- attack: для CurrentPlayer = 2
attack(GameState, AttackerIndex, TargetIndex, NewGameState) :-
    GameState = game_state(P1, P2, 2, Turn),
    P1 = player(H1, M1, MaxM1, Deck1, Hand1, Board1),
    P2 = player(H2, M2, MaxM2, Deck2, Hand2, Board2),
    nth1(AttackerIndex, Board2, minion(AttackerName, AttackerAttack, AttackerHealth, AttackerCanAttack, AttackerRushOnly)),
    ( AttackerCanAttack = true -> true ; fail ),
    ( AttackerRushOnly = true, TargetIndex = 0 -> fail ; true ),
    ( TargetIndex = 0 ->
        NewH1 is H1 - AttackerAttack,
        replace(AttackerIndex, Board2, minion(AttackerName, AttackerAttack, AttackerHealth, false, false), TempBoard2),
        NewP1 = player(NewH1, M1, MaxM1, Deck1, Hand1, Board1),
        NewP2 = player(H2, M2, MaxM2, Deck2, Hand2, TempBoard2),
        NewGameState = game_state(NewP1, NewP2, 2, Turn)
    ;
        nth1(TargetIndex, Board1, minion(TargetName, TargetAttack, TargetHealth, TargetCanAttack, TargetRushOnly)),
        NewAttackerHealth is AttackerHealth - TargetAttack,
        NewTargetHealth is TargetHealth - AttackerAttack,
        replace(AttackerIndex, Board2, minion(AttackerName, AttackerAttack, NewAttackerHealth, false, false), TempBoard2),
        replace(TargetIndex, Board1, minion(TargetName, TargetAttack, NewTargetHealth, TargetCanAttack, TargetRushOnly), TempBoard1),
        remove_dead_minions(TempBoard2, FinalBoard2),
        remove_dead_minions(TempBoard1, FinalBoard1),
        NewP2 = player(H2, M2, MaxM2, Deck2, Hand2, FinalBoard2),
        NewP1 = player(H1, M1, MaxM1, Deck1, Hand1, FinalBoard1),
        NewGameState = game_state(NewP1, NewP2, 2, Turn)
    ).

% end_turn / prepare_player_turn
% При end_turn передаём ход оппоненту, увеличиваем Turn и делаем prepare_player_turn для следующего игрока.
end_turn(GameState, NewGameState) :-
    GameState = game_state(P1, P2, CurrentPlayer, Turn),
    NextPlayer is 3 - CurrentPlayer,
    NextTurn is Turn + 1,
    ( CurrentPlayer = 1 ->
        prepare_player_turn(P2, NextTurn, NewP2),
        NewGameState = game_state(P1, NewP2, NextPlayer, NextTurn)
    ;
        prepare_player_turn(P1, NextTurn, NewP1),
        NewGameState = game_state(NewP1, P2, NextPlayer, NextTurn)
    ).

prepare_player_turn(player(H, _M, MaxM, Deck, Hand, Board), _Turn, NewPlayer) :-
    NewMaxM is min(10, MaxM + 1),
    NewMana = NewMaxM,
    draw_card(Deck, Hand, NewDeck, NewHand),
    set_minions_ready(Board, ReadyBoard),
    NewPlayer = player(H, NewMana, NewMaxM, NewDeck, NewHand, ReadyBoard).

% -----------------------
% ОТОБРАЖЕНИЕ
% -----------------------
% -----------------------
% Plain console display (no ANSI)
% -----------------------

% clear_screen: try platform clear (cls/clear); if fails, print blank lines
clear_screen :-
    ( current_prolog_flag(windows, true) ->
        ( catch(shell('cls'), _, fail) -> true ; forall(between(1,50,_), nl) )
    ;
        ( catch(shell('clear'), _, fail) -> true ; forall(between(1,50,_), nl) )
    ).

% Main plain display of game state: clears screen and prints info without ANSI
display_game(game_state(P1, P2, Current, Turn)) :-
    clear_screen,
    format('\n============ Hearthstone ============~n', []),
    format('Turn: ~w    Current player: P~w~n~n', [Turn, Current]),
    display_player_plain(1, P1, Current),
    nl,
    display_player_plain(2, P2, Current),
    format('=====================================~n~n', []).

% display_player_plain(+PlayerNumber, +PlayerTerm, +CurrentPlayer)
display_player_plain(Num, player(H, M, MaxM, Deck, Hand, Board), Current) :-
    ( Num =:= Current ->
        format('> Player ~w (CURRENT):~n', [Num])
    ;
        format('Player ~w:~n', [Num])
    ),
    format('  Health=~w  Mana=~w/~w~n', [H, M, MaxM]),
    % Hand
    length(Hand, HL),
    format('  Hand (~w): ', [HL]),
    display_hand_plain(Hand),
    nl,
    % Board
    length(Board, BL),
    format('  Board (~w):~n', [BL]),
    ( BL =:= 0 -> format('    (empty)~n') ; display_board_plain(Board) ),
    % Deck size
    length(Deck, DL),
    format('  Deck: ~w cards~n', [DL]).

% display_hand_plain(+HandList) - prints compact representation of cards in hand
display_hand_plain([]) :- format('(none)').
display_hand_plain([card(Name,Cost,Atk,Hp)]) :-
    format('[~w C:~w A:~w H:~w]', [Name, Cost, Atk, Hp]).
display_hand_plain([card(Name,Cost,Atk,Hp)|T]) :-
    format('[~w C:~w A:~w H:~w] ', [Name, Cost, Atk, Hp]),
    display_hand_plain(T).

% display_board_plain(+BoardList) prints indexed minions
display_board_plain(Board) :-
    display_board_plain(Board, 1).

display_board_plain([], _).
display_board_plain([Min|T], Index) :-
    display_minion_line_plain(Index, Min),
    Next is Index + 1,
    display_board_plain(T, Next).

% display_minion_line_plain(+Index, +minion(Name,Attack,Health,CanAttack,RushOnly))
display_minion_line_plain(Index, minion(Name,Atk,Hp,CanAttack,RushOnly)) :-
    ( CanAttack = true -> Ready = '[Ready]' ; Ready = '[Sleep]' ),
    ( RushOnly = true -> Rush = '[RushOnly]' ; Rush = '' ),
    format('    ~w) ~w  (ATK:~w  HP:~w)  ~w ~w~n', [Index, Name, Atk, Hp, Ready, Rush]).

% Fallback plain dump (keeps old behaviour if needed)
display_game_plain(State) :-
    write('Game state (plain):'), nl,
    write(State), nl.


% -----------------------
% ПРОВЕРКА ПОБЕДЫ
% -----------------------
check_win_condition(game_state(P1, P2, _, _), Winner) :-
    P1 = player(H1, _, _, _, _, _),
    P2 = player(H2, _, _, _, _, _),
    ( H1 =< 0 -> Winner = 2 ; H2 =< 0 -> Winner = 1 ; fail ).

% process_command wrapper (удобство для game_loop)
process_command(play(Minion), State, NewState) :- play_minion(State, Minion, NewState).
process_command(attack(Attacker, Target), State, NewState) :- attack(State, Attacker, Target, NewState).
process_command(end_turn, State, NewState) :- end_turn(State, NewState).

% game loop (консольный) 
game_loop(State) :-
    display_game(State),
    ( check_win_condition(State, Winner) ->
        format('Player ~w wins!~n', [Winner])
    ;
        format('Enter command (play(Name), attack(AttIdx, TargetIdx), end_turn):~n'),
        read(Command),
        ( process_command(Command, State, NewState) ->
            game_loop(NewState)
        ;
            format('Invalid command or cannot perform action.~n'),
            game_loop(State)
        )
    ).

start :- initial_state(State), game_loop(State).

% -----------------------
% ПРАВИЛА / ЛОГИЧЕСКИЕ ВЫВОДЫ
% -----------------------

% 1) strong_minion(Name) - сильный миньон (атк >=4 или HP >=5)
strong_minion(Name) :-
    ( card(Name, _, Attack, Health) -> true ; card_attack(Name, Attack), card_health(Name, Health) ),
    ( Attack >= 4 ; Health >= 5 ).

% 2) has_ability(Name, Ability)
has_ability(Name, Ability) :- ability_of(Name, Ability).

% 3) is_taunt(Name)
is_taunt(Name) :- ability_of(Name, taunt).

% 4) playable_card(PlayerAtom, CardName) - можно разыграть (использует in_hand/2 и player_mana/2)
% Примерная модель: player_mana/2 даёт текущую доступную ману для "атома" игрока
player_mana(player1, 1).
player_mana(player2, 1).

playable_card(PlayerAtom, CardName) :-
    in_hand(PlayerAtom, CardName),
    card_cost(CardName, Cost),
    player_mana(PlayerAtom, Mana),
    Mana >= Cost.

% 5) vulnerable_minion(Name) - миньон с HP <= 2
vulnerable_minion(Name) :-
    card_health(Name, H),
    H =< 2.

% 6) can_attack_immediately(Name) - charge или rush
can_attack_immediately(Name) :- has_ability(Name, charge).
can_attack_immediately(Name) :- has_ability(Name, rush).

% 7) total_board_attack(PlayerAtom, SumAttack) - сумма атак всех миньонов на доске игрока
total_board_attack(PlayerAtom, Sum) :-
    findall(Attack, (in_board(PlayerAtom, MinionName), card_attack(MinionName, Attack)), Attacks),
    sum_list(Attacks, Sum).

% 8) lethal_possible(PlayerAtom, OppHealth) - упрощённая проверка
lethal_possible(PlayerAtom, OppHealth) :-
    total_board_attack(PlayerAtom, SumAtk),
    SumAtk >= OppHealth.


card_of_tribe(Tribe, CardName) :- tribe_of(CardName, Tribe).
