// @ts-check
const path = require('path');

const repo = 'https://github.com/afdEP/afdEP-comm';

// Prism themes limited to the logo palette (colors match src/css/custom.css).
const prism = (fg, bg, muted, accent) => ({
  plain: {color: fg, backgroundColor: bg},
  styles: [
    {types: ['comment', 'prolog'], style: {color: muted, fontStyle: 'italic'}},
    {types: ['keyword', 'builtin', 'function', 'boolean', 'number'], style: {color: accent}},
    {types: ['string', 'char', 'attr-value'], style: {color: muted}},
    {types: ['punctuation', 'operator'], style: {color: muted}},
  ],
});
const prismDark = prism('#FFFFFF', '#000000', '#9E9D9D', '#FCBA00');
const prismLight = prism('#000000', '#FFFFFF', '#5F5E5E', '#8A6400');

/** @type {import('@docusaurus/types').Config} */
module.exports = {
  title: 'afdEP-comm',
  tagline: 'Expert-parallel communication for Attention-FFN Disaggregation',
  url: 'https://afdep.github.io',
  // PR previews build under pr-preview/pr-N/ (.github/workflows/docs.yml).
  baseUrl: process.env.BASE_URL || '/afdEP-comm/',
  organizationName: 'afdEP',
  projectName: 'afdEP-comm',
  trailingSlash: false,
  onBrokenLinks: 'throw',
  staticDirectories: ['../docs/assets'],
  markdown: {
    format: 'detect',
    hooks: {
      // Links that leave docs/ (README, examples, ...) point at the repository.
      onBrokenMarkdownLinks: ({sourceFilePath, url}) =>
        `${repo}/blob/main/${path.posix.join('website', path.posix.dirname(sourceFilePath), url)}`,
    },
  },
  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {path: '../docs', routeBasePath: 'docs'},
        blog: false,
        theme: {customCss: './src/css/custom.css'},
      }),
    ],
  ],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Dark unless the reader switches; the choice is kept in the browser.
      colorMode: {defaultMode: 'dark', respectPrefersColorScheme: false},
      navbar: {
        logo: {alt: 'afdEP-comm', src: 'afdep-logo.png'},
        items: [
          {type: 'docSidebar', sidebarId: 'defaultSidebar', label: 'Docs', position: 'left'},
          {href: repo, label: 'GitHub', position: 'right'},
        ],
      },
      prism: {theme: prismLight, darkTheme: prismDark, additionalLanguages: ['bash']},
    }),
};
