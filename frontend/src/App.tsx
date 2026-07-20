import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { EnrollmentPage } from "./pages/EnrollmentPage";
import { InferencePage } from "./pages/InferencePage";
import { ProfilesPage } from "./pages/ProfilesPage";

export function App() {
  return <AppShell><Routes><Route path="/profiles" element={<ProfilesPage />} /><Route path="/enroll/:profileId" element={<EnrollmentPage />} /><Route path="/inference" element={<InferencePage />} /><Route path="*" element={<Navigate to="/profiles" replace />} /></Routes></AppShell>;
}
