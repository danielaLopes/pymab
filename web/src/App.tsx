import { Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "./components/game";
import { RuntimeProvider } from "./engine/RuntimeProvider";
import { HomeRoute } from "./routes/HomeRoute";
import { LabRoute } from "./routes/LabRoute";
import { LessonRoute } from "./routes/LessonRoute";

export function App() {
  return (
    <AppShell>
      <RuntimeProvider>
        <Routes>
          <Route path="/" element={<HomeRoute />} />
          <Route path="/lesson/:lessonSlug" element={<LessonRoute />} />
          <Route path="/lab" element={<LabRoute />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </RuntimeProvider>
    </AppShell>
  );
}
