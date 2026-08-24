import { CampaignMap, LessonBadge } from "../components/game";

export function HomeRoute() {
  return (
    <main className="home-page">
      <section className="hero">
        <div>
          <p className="eyebrow">Interactive bandit lessons · Powered by real Python</p>
          <h1>
            Learn the art of choosing <em>before</em> you know.
          </h1>
          <p className="hero-copy">
            Enter a sequence of independent crossroads. Watch PyMAB explore, learn, and balance
            uncertainty—one transparent decision at a time.
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
            <p className="eyebrow">Your expedition</p>
            <h2 id="choose-mission">Choose a mission</h2>
          </div>
          <p>Each journey begins guided, becomes a challenge, then opens into a full simulation.</p>
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
            <strong>Read the chamber</strong>
            <p>See what the policy knows before it acts.</p>
          </li>
          <li>
            <span>02</span>
            <strong>Watch the choice</strong>
            <p>The checked-out PyMAB wheel selects a gate.</p>
          </li>
          <li>
            <span>03</span>
            <strong>Reveal and learn</strong>
            <p>Observe reward, regret, and the exact update.</p>
          </li>
        </ol>
      </section>
    </main>
  );
}
