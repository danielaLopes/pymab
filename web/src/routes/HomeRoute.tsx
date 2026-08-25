import { CampaignMap, LessonBadge } from "../components/game";

export function HomeRoute() {
  return (
    <main className="home-page">
      <section className="hero">
        <div>
          <p className="eyebrow">Interactive bandit lessons · Powered by PyMAB</p>
          <h1>
            See how bandit algorithms choose with <em>incomplete information</em>.
          </h1>
          <p className="hero-copy">
            Run a sequence of independent decisions. Watch PyMAB balance estimated reward against
            uncertainty, one choice at a time.
          </p>
        </div>
        <div className="hero-orbit" aria-hidden="true">
          <LessonBadge lessonId="epsilon-greedy" />
          <span className="orbit-ring" />
          <LessonBadge lessonId="linucb" />
        </div>
      </section>
      <section aria-labelledby="choose-mission">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Lessons</p>
            <h2 id="choose-mission">Choose a mission</h2>
          </div>
          <p>
            Start with a guided lesson, test yourself in challenge mode, then adjust the simulation
            in free play.
          </p>
        </div>
        <CampaignMap />
      </section>
      <section className="how-it-works" aria-labelledby="how-heading">
        <div>
          <p className="eyebrow">The loop</p>
          <h2 id="how-heading">A small game with a real algorithm inside.</h2>
        </div>
        <ol>
          <li>
            <span>01</span>
            <strong>Review the inputs</strong>
            <p>See what information the policy has before it acts.</p>
          </li>
          <li>
            <span>02</span>
            <strong>Watch the choice</strong>
            <p>The PyMAB policy selects a gate.</p>
          </li>
          <li>
            <span>03</span>
            <strong>Check the result</strong>
            <p>Review the reward, regret, and exact policy update.</p>
          </li>
        </ol>
      </section>
    </main>
  );
}
