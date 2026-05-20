import "./styles.css";

export const metadata = {
  title: "AlignGPT",
  description: "Alignment research and operations dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
