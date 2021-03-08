use proconio::input;
use std::time::Instant;
use std::cmp;
use std::fmt;
use Direction::*;

const TIME_LIMIT: u128 = 4900;
const LOOP_PER_TIME_CHECK: usize = 100;
const START_TMP: f64 = 0.001;
const END_TMP: f64 = 0.0001;
const SZ: i64 = 10000;

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
    n: usize,
    rand: Xorshift,
    score: f64,
    score_v: Vec<f64>,
    x: &'a Vec<i64>,
    y: &'a Vec<i64>,
    r: &'a Vec<i64>,
    adv: Vec<Advertizement>,
    dir: Vec<Direction>,
    cntupd: i64,
    cntchal: i64,
    midiff: f64,
    madiff: f64,
    upd_idx: Vec<i64>,
}
impl<'a> State<'a> {
    fn new(n: usize, rand: Xorshift, x: &'a Vec<i64>, y: &'a Vec<i64>, r: &'a Vec<i64>) -> Self {
        let mut state = State{
            n: n,
            rand: rand,
            score: 0.0,
            score_v: vec![0.0; n],
            x: x,
            y: y,
            r: r,
            adv: vec![Advertizement{x1:-1, y1:-1, x2:-1, y2:-1} ;n],
            dir: vec![Left, Right, Up, Down],
            cntupd: 0,
            cntchal: 0,
            midiff: 100000000.0,
            madiff: 0.0,
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

    fn update(&mut self, incr: bool, annealing: bool, score_prob: bool, temperature: f64, val: i64) {
        // 長方形ごとのスコアに応じて確率を計算
        let mut i = self.rand.rand_int(0, (self.n-1) as u64) as usize;
        if score_prob {
            let mut p: Vec<f64> = vec![0.0; self.n+1];
            let mut su = 0.0;
            for i in 0..self.n {
                let val = self.score_v[i].powi(2);
                su += val;
                p[i+1] = val;
            }
            for i in 1..self.n+1 {
                p[i] /= su;
                if i != 0 {
                    p[i] += p[i-1];
                }
            }
            // 変化させるidx
            let r = self.rand.randf();
            for j in 0..self.n {
                if p[j] <= r && r < p[j+1] {
                    i = j;
                    break;
                }
            }
        }
        self.upd_idx[i] += 1;
        // 変化させる方向
        // 1: ひろげる, 2: ちぢめる
        let sign: i64 = 
            if incr {
                1
            } else {
                self.rand.rand_int(0, 1) as i64 * 2 - 1
            };

        let dir_idx = self.rand.rand_int(0, 3) as usize;
        // どの辺を操作するか
        let dir = self.dir[dir_idx].clone();

        // 拡張した時に重なった長方形
        let mut shrinked: Vec<usize> = Vec::new();
        self.score_v[i] -= self.score(i);
        let mut ok = self.update_adv(i, &dir, sign, val, &mut shrinked);
        self.score_v[i] += self.score(i);


        // shrinkedの各要素を縮める
        if ok {
            for j in 0..shrinked.len() {
                let ndir = dir.reverse();
                let mut fake_shrinked: Vec<usize> = Vec::new();
                self.score_v[shrinked[j]] -= self.score(shrinked[j]);
                ok &= self.update_adv(shrinked[j], &ndir, -1, val, &mut fake_shrinked);
                self.score_v[shrinked[j]] += self.score(shrinked[j]);
                assert_eq!(fake_shrinked.len(), 0);
            }
        }
        /*
        let mut new_score = 0.0;
        for i in 0..self.n {
            new_score += self.score_v[i];
        }
        */
        let new_score = self.score_all();
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
                //eprintln!("{} : {} : {}", prob, temperature, new_score - self.score);
                self.rand.randf() < prob
            } else {
                // 山登りの時
                new_score > self.score
            };

        if upd {
            self.cntupd += 1;
            self.score = new_score;
        } else {
            self.revert(i, &dir, sign, val, &mut shrinked);
        }
    }

    // 10^9をかける手前までのスコアを計算
    fn score_all(&mut self) -> f64 {
        let mut score: f64 = 0.0;
        for i in 0..self.n {
            if self.adv[i].x1 == -1 {
                continue;
            }
            score += self.score(i) as f64;
        }
        score
    }

    // 広告ひとつのスコアを計算
    fn score(&self, i: usize) -> f64 {
        // 1 - (1 - min(ri, si) / max(ri, si)) ^ 2
        let s = self.adv[i].area();
        1.0 - (1.0 - cmp::min(self.r[i], s) as f64 / cmp::max(self.r[i], s) as f64).powi(2)
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
        self.score_v[i] -= self.score(i);
        match dir {
            Left => self.adv[i].x1 += sign * val,
            Right => self.adv[i].x2 -= sign * val,
            Up => self.adv[i].y1 += sign * val,
            Down => self.adv[i].y2 -= sign * val,
        }
        self.score_v[i] += self.score(i);
        let ndir = dir.reverse();
        let mut fake_shrinked = Vec::new();
        for j in 0..shrinked.len() {
            self.revert(shrinked[j], &ndir, -1, val, &mut fake_shrinked);
        }
    }
}

fn simulate(state: &mut State, start: &Instant, time_limit: u128, incr: bool, annealing: bool, score_prob: bool, val: i64) {
    let mut elapsed_time = start.elapsed().as_millis();
    while elapsed_time < time_limit {
        let temperature: f64 = START_TMP + (END_TMP - START_TMP) * (elapsed_time as f64) / (TIME_LIMIT as f64);
        for _ in 0..LOOP_PER_TIME_CHECK {
            state.update(incr, annealing, score_prob, temperature, val);
        }
        elapsed_time = start.elapsed().as_millis();
    }
}

fn main() {
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
    
    let seed = start.elapsed().as_nanos() as u64;
    let rand = Xorshift::with_seed(seed);
    let mut state = State::new(n, rand, &x, &y, &r);
    simulate(&mut state, &start, TIME_LIMIT / 30, true, true, true, 100);
    simulate(&mut state, &start, TIME_LIMIT / 3 * 2, false, true, false, 10);
    simulate(&mut state, &start, TIME_LIMIT / 5 * 4, true, false, true, 10);


    let mul = 1000000000;
    eprintln!("cntchal: {}", state.cntchal);
    eprintln!("cntupd: {}", state.cntupd);
    eprintln!("midiff: {}", state.midiff);
    eprintln!("madiff: {}", state.madiff);
    eprintln!("saved score: {}", state.score / n as f64 * mul as f64);
    eprintln!("calculated score: {}", state.score_all() / n as f64 * mul as f64);
    /*
    eprintln!("upd_idx");
    for i in 0..n {
        eprintln!("{} : {}", i, state.upd_idx[i]);
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

}