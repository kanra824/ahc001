use proconio::input;
use std::time::Instant;
use std::cmp;
use std::fmt;
use std::fs::File;
use std::io::{Write};
use Direction::*;

const TIME_LIMIT: u128 = 4900;
const LOOP_PER_TIME_CHECK: usize = 100;
const SZ: i64 = 10000;
const OUTPUT_NUM: u128 = 3;

#[derive(Clone)]
pub enum Direction {
    Left,
    Right,
    Up,
    Down,
}
impl Direction {
    fn reverse(&self) -> Direction {
        match self {
            Left => Right,
            Right => Left,
            Up => Down,
            Down => Up
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct Xorshift {
    seed: u64,
}
impl Xorshift {
    #[allow(dead_code)]
    pub fn new() -> Xorshift {
        Xorshift {
            seed: 0xf0fb588ca2196dac,
        }
    }
    #[allow(dead_code)]
    pub fn with_seed(seed: u64) -> Xorshift {
        Xorshift { seed: seed }
    }
    #[inline]
    #[allow(dead_code)]
    pub fn next(&mut self) -> u64 {
        self.seed = self.seed ^ (self.seed << 13);
        self.seed = self.seed ^ (self.seed >> 7);
        self.seed = self.seed ^ (self.seed << 17);
        self.seed
    }
    #[inline]
    #[allow(dead_code)]
    pub fn rand(&mut self, m: u64) -> u64 {
        self.next() % m
    }
    #[inline]
    #[allow(dead_code)]
    pub fn rand_int(&mut self, min: u64, max: u64) -> u64 {
        self.next() % (max - min + 1) + min
    }
    #[inline]
    #[allow(dead_code)]
    // 0.0 ~ 1.0
    pub fn randf(&mut self) -> f64 {
        use std::mem;
        const UPPER_MASK: u64 = 0x3FF0000000000000;
        const LOWER_MASK: u64 = 0xFFFFFFFFFFFFF;
        let tmp = UPPER_MASK | (self.next() & LOWER_MASK);
        let result: f64 = unsafe { mem::transmute(tmp) };
        result - 1.0
    }
}

#[derive(Clone)]
struct Advertizement {
    x1: i64,
    y1: i64,
    x2: i64,
    y2: i64,
}
impl Advertizement {
    fn area(&self) -> i64 {
        (self.x1 - self.x2).abs() * (self.y1 - self.y2).abs()
    }

    fn cross(&self, adv: &Advertizement) -> bool {
        let mut res = true;
        res = res && ((self.x1 <= adv.x1 && adv.x1 < self.x2) || (adv.x1 < self.x1 && self.x1 <= adv.x2));
        res = res && ((self.y1 <= adv.y1 && adv.y1 < self.y2) || (adv.y1 < self.y1 && self.y1 <= adv.y2));
        res
    }
}
impl fmt::Display for Advertizement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {} {} {}", self.x1, self.y1, self.x2, self.y2)
    }
}

struct State<'a> {
    start_tmp: f64,
    end_tmp: f64,
    n: usize,
    rand: Xorshift,
    score: f64,
    score_v: Vec<f64>,
    prob_v: Vec<f64>,
    prob_sum: f64,
    x: &'a Vec<i64>,
    y: &'a Vec<i64>,
    r: &'a Vec<i64>,
    adv: Vec<Advertizement>,
    dir: Vec<Direction>,
    cntupd: i64,
    cntchal: i64,
    midiff: f64,
    madiff: f64,
    chal_idx: Vec<i64>,
    upd_idx: Vec<i64>,
}
impl<'a> State<'a> {
    fn new(n: usize, rand: Xorshift, x: &'a Vec<i64>, y: &'a Vec<i64>, r: &'a Vec<i64>, start_tmp: f64, end_tmp: f64) -> Self {
        let mut state = State{
            n: n,
            start_tmp: start_tmp,
            end_tmp: end_tmp,
            rand: rand,
            score: 0.0,
            score_v: vec![0.0; n],
            prob_v: vec![0.0; n],
            prob_sum: 0.0,
            x: x,
            y: y,
            r: r,
            adv: vec![Advertizement{x1:-1, y1:-1, x2:-1, y2:-1} ;n],
            dir: vec![Left, Right, Up, Down],
            cntupd: 0,
            cntchal: 0,
            midiff: 100000000.0,
            madiff: 0.0,
            chal_idx: vec![0; n],
            upd_idx: vec![0; n],
        };
        for i in 0..n {
            state.adv[i] = Advertizement{
                x1: x[i],
                y1: y[i],
                x2: x[i]+1,
                y2: y[i]+1,
            };
        }
        state.score = state.score_all();
        state
    }

    fn update(&mut self, mut sign: i64, annealing: bool, score_prob: bool, temperature: f64, val: i64) {
        // 長方形ごとのスコアに応じて確率を計算
        let mut i = self.rand.rand_int(0, (self.n-1) as u64) as usize;
        if score_prob {
            // 変化させるidx
            let r = self.rand.randf();
            let mut now = 0.0;
            for j in 0..self.n {
                if now <= r && r < now + self.prob_v[j] / self.prob_sum {
                    i = j;
                    break;
                }
                now += self.prob_v[j] / self.prob_sum;
            }
        }
        
        let dir_idx = self.rand.rand_int(0, 3) as usize;
        let dir_idx2 = self.rand.rand_int(0, 3) as usize;
        // どの辺を操作するか
        let dir = self.dir[dir_idx].clone();
        let dir2 = self.dir[dir_idx2].clone();

        // 変化させる方向
        // 1: ひろげる, -1: ちぢめる
        // 指定がなければ確率でちぢめてひろげる
        if sign == 0 {
            let p = self.rand.randf();
            sign = if p < 0.4 {
                1
            } else if p < 0.4 {
                -1
            } else {
                0
            }
        }
        // 拡張した時に重なった長方形
        let mut shrinked: Vec<usize> = Vec::new();
        let mut ok;
        if sign == 1 {
            ok = self.update_adv(i, &dir, 1, val, &mut shrinked);
        } else if sign == -1 {
            ok = self.update_adv(i, &dir, -1, val, &mut shrinked);
        } else {
            ok = self.update_adv(i, &dir2, -1, val, &mut shrinked);
            assert_eq!(shrinked.len(), 0);
            ok &= self.update_adv(i, &dir, 1, val, &mut shrinked);
        }
        let mut new_score = self.score;
        new_score -= self.score_v[i];
        self.score_v[i] = self.score(i);
        new_score += self.score_v[i];
        self.prob_sum -= self.prob_v[i];
        self.prob_v[i] = self.calc_prob(i);
        self.prob_sum += self.prob_v[i];


        // shrinkedの各要素を縮める
        for j in &shrinked {
            let idx = *j;
            let ndir = dir.reverse();
            let mut fake_shrinked: Vec<usize> = Vec::new();
            ok &= self.update_adv(idx, &ndir, -1, val, &mut fake_shrinked);
            
            // スコアの計算と更新
            new_score -= self.score_v[idx];
            self.score_v[idx] = self.score(idx);
            new_score += self.score_v[idx];

            // 確率の計算と更新
            self.prob_sum -= self.prob_v[idx];
            self.prob_v[idx] = self.calc_prob(idx);
            self.prob_sum += self.prob_v[idx];
            assert_eq!(fake_shrinked.len(), 0);
        }
        let diff = new_score - self.score;

        self.cntchal += 1;
        if annealing {
            if diff < 0.0 {
                self.madiff = self.madiff.max(diff);
            }
            self.midiff = self.midiff.min(diff);
        }
        let upd = ok &&
            if annealing {
                // 焼きなましの時
                let prob = std::f64::consts::E.powf((new_score - self.score) as f64 / temperature);
                //eprintln!("{} : {} : {} : {}", i, prob, temperature, new_score - self.score);
                self.rand.randf() < prob
            } else {
                // 山登りの時
                new_score > self.score
            };

        self.chal_idx[i] += 1;
        if upd {
            self.upd_idx[i] += 1;
            self.cntupd += 1;
            self.score = new_score;
        } else {
            if sign == 1 {
                self.revert(i, &dir, 1, val, &shrinked);
            } else if sign == -1 {
                self.revert(i, &dir, -1, val, &shrinked);
            } else {
                let fake_shrinked = Vec::new();
                self.revert(i, &dir, 1, val, &shrinked);
                self.revert(i, &dir2, -1, val, &fake_shrinked);
            }



        }
    }

    // 10^9をかける手前までのスコアを計算
    fn score_all(&mut self) -> f64 {
        let mut score: f64 = 0.0;
        for i in 0..self.n {
            if self.adv[i].x1 == -1 {
                continue;
            }
            let nowscore = self.score(i) as f64;
            score += nowscore;
            self.score_v[i] = nowscore;
            self.prob_sum -= self.prob_v[i];
            self.prob_v[i] = self.calc_prob(i);
            self.prob_sum += self.prob_v[i];
        }
        score
    }

    // 広告ひとつのスコアを計算
    fn score(&self, i: usize) -> f64 {
        // 1 - (1 - min(ri, si) / max(ri, si)) ^ 2
        let s = self.adv[i].area();
        1.0 - (1.0 - cmp::min(self.r[i], s) as f64 / cmp::max(self.r[i], s) as f64).powi(2)
    }

    // idxを選択する確率に用いる、正規化する前の値
    fn calc_prob(&self, i: usize) -> f64 {
        self.score_v[i].powi(2)
        /*
        if self.adv[i].area() - self.r[i] > 0 {
            0.0
        } else {
            1.0 / (self.score_v[i] + 0.0000001)
        }
        */
    }

    fn update_adv(&mut self, i: usize, dir: &Direction, sign: i64, val: i64, shrinked: &mut Vec<usize>) -> bool {
        // i番目の広告を更新
        match dir {
            Left => self.adv[i].x1 -= sign * val,
            Right => self.adv[i].x2 += sign * val,
            Up => self.adv[i].y1 -= sign * val,
            Down => self.adv[i].y2 += sign *val,
        }

        // 更新後の長方形が(単独で)正しいかどうか確認
        let ok = match dir {
            Left => 0 <= self.adv[i].x1 && self.adv[i].x1 <= self.x[i],
            Right => self.x[i] < self.adv[i].x2 && self.adv[i].x2 <= SZ,
            Up => 0 <= self.adv[i].y1 && self.adv[i].y1 <= self.y[i],
            Down => self.y[i] < self.adv[i].y2 && self.adv[i].y2 <= SZ,
        };

        if ok && sign == 1 {
            // 更新後の長方形と重なる長方形のidxをshrinkedに格納
            for j in 0..self.n {
                if i != j && self.adv[i].cross(&self.adv[j]) {
                    shrinked.push(j);
                }
            }
        }

        ok
    }

    // dir, sign, valはupdate_advの時と同じ値であることに注意
    fn revert(&mut self, i: usize, dir: &Direction, sign: i64, val: i64, shrinked: &Vec<usize>) {
        match dir {
            Left => self.adv[i].x1 += sign * val,
            Right => self.adv[i].x2 -= sign * val,
            Up => self.adv[i].y1 += sign * val,
            Down => self.adv[i].y2 -= sign * val,
        }
        self.score_v[i] = self.score(i);
        self.prob_sum -= self.prob_v[i];
        self.prob_v[i] = self.calc_prob(i);
        self.prob_sum += self.prob_v[i];
        let ndir = dir.reverse();
        let mut fake_shrinked = Vec::new();
        for j in shrinked {
            self.revert(*j, &ndir, -1, val, &mut fake_shrinked);
        }
    }
}

fn _simulate(state: &mut State, start: &Instant, time_limit: u128, sign: i64, annealing: bool, score_prob: bool, val: i64) {
    let mut elapsed_time = start.elapsed().as_millis();
    while elapsed_time < time_limit {
        let temperature: f64 = state.start_tmp + (state.end_tmp - state.start_tmp) * (elapsed_time as f64) / (TIME_LIMIT as f64);
        for _ in 0..LOOP_PER_TIME_CHECK {
            state.update(sign, annealing, score_prob, temperature, val);
        }
        elapsed_time = start.elapsed().as_millis();
    }
}

fn simulate_with_output(state: &mut State, start: &Instant, time_limit: u128, sign: i64, annealing: bool, score_prob: bool, val: i64, num: &String, pos: &mut u128) -> Result<(), Box<dyn std::error::Error>> {
    let mut elapsed_time = start.elapsed().as_millis();
    while elapsed_time < time_limit {
        let temperature: f64 = state.start_tmp + (state.end_tmp - state.start_tmp) * (elapsed_time as f64) / (TIME_LIMIT as f64);
        for _ in 0..LOOP_PER_TIME_CHECK {
            state.update(sign, annealing, score_prob, temperature, val);
        }
        elapsed_time = start.elapsed().as_millis();

        if elapsed_time > TIME_LIMIT / OUTPUT_NUM * *pos {
            let path = format!("./tester/tools/out/{}/{}.txt", num, pos);
            let mut file = File::create(path)?;
            for adv in &state.adv {
                writeln!(file, "{}", adv)?;
            }
            *pos += 1;
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();

    input! {
        in_n: usize,
        input: [[i64; 3]; in_n],
    }

    let mut inx: Vec<i64> = vec![0; in_n];
    let mut iny: Vec<i64> = vec![0; in_n];
    let mut inr: Vec<i64> = vec![0; in_n];

    for i in 0..in_n {
        inx[i] = input[i][0];
        iny[i] = input[i][1];
        inr[i] = input[i][2];
    }

    let mut sel = vec![true; in_n];

    for i in 0..in_n {
        for j in 0..i {
            if inx[i] == inx[j] && iny[i] == iny[j] {
                if inr[i] <= inr[j] {
                    sel[j] = false;
                } else {
                    sel[i] = false;
                }
            }
        }
    }

    let mut x: Vec<i64> = Vec::new();
    let mut y: Vec<i64> = Vec::new();
    let mut r: Vec<i64> = Vec::new();
    let mut sel_idx: Vec<usize> = Vec::new();
    let mut not_sel_idx: Vec<usize> = Vec::new();

    for i in 0..in_n {
        if sel[i] {
            x.push(inx[i]);
            y.push(iny[i]);
            r.push(inr[i]);
            sel_idx.push(i);
        } else {
            not_sel_idx.push(i);
        }
    }

    let n = x.len();

    let (start_time, end_time) =
    if std::env::args().len() >= 2 {
        (
            std::env::args().nth(1).unwrap().parse().unwrap(),
            std::env::args().nth(2).unwrap().parse().unwrap()
        )
    } else {
        //(0.008836644575520086, 0.008950549607649214) // optuna
        (0.0006979039523455251, 0.01290817137136288) // 479
        //(0.001, 0.0001)
    };
    
    let seed = start.elapsed().as_nanos() as u64;
    let rand = Xorshift::with_seed(seed);
    let mut state = State::new(n, rand, &x, &y, &r, start_time, end_time);
    let mut pos = 0;
    let num = "0000".to_string();
    simulate_with_output(&mut state, &start, TIME_LIMIT / 30, 1, false, false, 100, &num, &mut pos)?;
    simulate_with_output(&mut state, &start, TIME_LIMIT / 10 * 9, 0, true, false, 10, &num, &mut pos)?;
    simulate_with_output(&mut state, &start, TIME_LIMIT, 0, false, true, 10, &num, &mut pos)?;


    let mul = 1000000000;
    eprintln!("cntchal: {}", state.cntchal);
    eprintln!("cntupd: {}", state.cntupd);
    //eprintln!("midiff: {}", state.midiff);
    //eprintln!("madiff: {}", state.madiff);
    eprintln!("saved score: {}", state.score / n as f64 * mul as f64);
    eprintln!("calculated score: {}", state.score_all() / n as f64 * mul as f64);
    /*
    for i in 0..n {
        eprintln!("{} : {} : {} : {}", i, state.upd_idx[i], state.chal_idx[i], state.score_v[i]);
    }
    */


    // print answer
    if n == in_n {
        for i in 0..n {
            println!("{}", state.adv[i]);
        }
    } else {
        panic!("入力に同じ(x, y)が存在")
    }

    Ok(())
}